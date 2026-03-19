

using KomaMRICore, KomaMRIPlots, KomaMRIFiles, CUDA, PlotlyJS # Essentials

using Suppressor, ProgressLogging # Extras

sys = Scanner()
sys.B0 = 0.55
sys.Gmax = 40.0e-3
sys.Smax = 25.0


Trf = 500e-6  			# 500 [ms]
B1 = 1 / (360*γ*Trf)    # B1 amplitude [uT]
Tadc = 1e-6 			# 1us

	# Prepulses
Tfatsat = 26.624e-3     # 26.6 [ms]
T2prep_duration = 50e-3 # 50 [ms]

	# Acquisition
RR = 0.9 				# 1 [s]
dummy_heart_beats = 3 	# Steady-state
TR = 6.27e-3             # 5.3 [ms] RF Low SAR
TE = TR / 2 			# bSSFP condition
iNAV_lines = 3          # FatSat-Acq delay: iNAV_lines * TR
iNAV_flip_angle = 3.2   # 3.2 [deg]
im_segments = 30        # Acquisitino window: im_segments * TR
	# To be optimized
im_flip_angle = 110    # 110 [deg]
FatSat_flip_angle = 180 # 180 [deg]

#t2p_50 = read_seq("examples/4.reproducible_notebooks/t2_mlev4_50.seq") # Pulseq import

seq_params = (;
	dummy_heart_beats,
	iNAV_lines,
	im_segments,
	iNAV_flip_angle,
	im_flip_angle,
	#t2p_50,
	T2prep_duration,
	FatSat_flip_angle,
	RR
)
seq_params

fat_ppm = -3.4e-6 			# -3.4ppm fat-water frequency shift
Niso = 200        			# 200 isochromats in spoiler direction
Δx_voxel = 0.96e-3 			# 1.5 [mm]
fat_freq = γ*sys.B0*fat_ppm # -80 [Hz]
dx = Array(range(-Δx_voxel/2, Δx_voxel/2, Niso))
#t2p_50

function FatSat(α, Δf; sample=false)
	    # FatSat design
		# cutoff_freq = sqrt(log(2) / 2) / a where B1(t) = exp(-(π t / a)^2)
	cutoff = fat_freq / π 			      # cutoff [Hz] => ≈1/10 RF power to water
	a = sqrt(log(2) / 2) / cutoff         # a [s]
	τ = range(-Tfatsat/2, Tfatsat/2, 64) # time [s]
	gauss_pulse = exp.(-(π * τ / a) .^ 2) # B1(t) [T]
		# FatSat prepulse
    seq = Sequence()
    seq += Grad(-8e-3, 3000e-6, 500e-6) #Spoiler1
    seq += RF(gauss_pulse, Tfatsat, Δf)
    α_ref = get_flip_angles(seq)[2]
    seq *= (α/α_ref+0im)
    if sample
        seq += ADC(1, 1e-6)
    end
    seq += Grad(8e-3, 3000e-6, 500e-6) #Spoiler2
    if sample
        seq += ADC(1, 1e-6)
    end
    return seq
end

function T2prep(duration; sample=false)
    # For T2prep, the effective TE is approximately duration - 3*Trf
    # We use this to achieve the desired total duration
    TE = duration - 3*Trf
    if TE <= 0
        error("T2prep duration too short, minimum duration is approximately $(3*Trf*1e3) ms")
    end
    seq = Sequence()
    seq += RF(90 * B1, Trf)
    seq += sample ? ADC(20, TE/2 - 1.5Trf) : Delay(TE/2 - 1.5Trf)
    seq += RF(180im * B1 / 2, Trf*2)
    seq += sample ? ADC(20, TE/2 - 1.5Trf) : Delay(TE/2 - 1.5Trf)
    seq += RF(-90 * B1, Trf)
    seq += Grad(8e-3, 6000e-6, 600e-6)
    if sample
        seq += ADC(1, 1e-6)
    end
    return seq
end

function bSSFP(iNAV_lines, im_segments, iNAV_flip_angle, im_flip_angle; sample=false)
    k = 0
    seq = Sequence()
    for i = 0 : iNAV_lines + im_segments - 1
        if iNAV_lines != 0
            m = (im_flip_angle - iNAV_flip_angle) / iNAV_lines
            α = min( m * i + iNAV_flip_angle, im_flip_angle ) * (-1)^k
        else
            α = im_flip_angle * (-1)^k
        end
        seq += RF(α * B1, Trf)
        if i < iNAV_lines && !sample
            seq += Delay(TR - Trf)
        else
            seq += Delay(TE - Trf/2 - Tadc/2)
            seq += ADC(1, Tadc)
            seq += Delay(TR - TE - Tadc/2 - Trf/2)
        end
        k += 1
    end
    return seq
end

function CMRA(
			dummy_heart_beats,
			iNAV_lines,
			im_segments,
			iNAV_flip_angle,
			im_flip_angle,
			#t2p_50,
			T2prep_duration,
			FatSat_flip_angle=180,
			RR=0.9;
			sample_recovery=zeros(Bool, dummy_heart_beats+1)
			)
	# Seq init
	seq = Sequence()
	for hb = 1 : dummy_heart_beats + 1
		sample = sample_recovery[hb] # Sampling recovery curve for hb
			# Generating seq blocks
		   #t2p = t2p_50
		t2p = T2prep(T2prep_duration; sample)
		#t2p = T2prep(t2p_50; sample)
	    fatsat = FatSat(FatSat_flip_angle, fat_freq; sample)
        bssfp = bSSFP(iNAV_lines, im_segments, iNAV_flip_angle, im_flip_angle; sample)
        # Concatenating seq blocks
        seq += t2p
        seq += fatsat
        seq += bssfp
		# RR interval consideration
		RRdelay = RR  - dur(bssfp) - dur(t2p) - dur(fatsat)
        seq += sample ? ADC(80, RRdelay) : Delay(RRdelay)
    end
    return seq
end

function cardiac_phantom(off; off_fat=fat_freq)
	myocard = Phantom(x=dx, ρ=0.6*ones(Niso), T1=750e-3*ones(Niso),
                               T2=90e-3*ones(Niso),    Δw=2π*off*ones(Niso))
    blood =   Phantom(x=dx, ρ=0.7*ones(Niso), T1=1122e-3*ones(Niso),
                               T2=263e-3*ones(Niso),   Δw=2π*off*ones(Niso))
    fat1 =    Phantom(x=dx, ρ=1.0*ones(Niso), T1=183e-3*ones(Niso),
                               T2=93e-3*ones(Niso),    Δw=2π*(off_fat + off)*ones(Niso))
    #fat2 =    Phantom(x=dx, ρ=1.0*ones(Niso), T1=130e-3*ones(Niso),
                               #T2=93e-3*ones(Niso),    Δw=2π*(off_fat + off)*ones(Niso))
    obj = myocard + blood + fat1
    return obj
end

sim_params = Dict{String,Any}(
	"return_type"=>"mat",
	"sim_method"=>BlochDict(save_Mz=true),
	"Δt_rf"=>Trf,
	"gpu"=>false,
	"Nthreads"=>1
)

seq = CMRA(seq_params...; sample_recovery=ones(Bool, dummy_heart_beats+1))
obj = cardiac_phantom(0)
magnetization = @suppress simulate(obj, seq, sys; sim_params=sim_params)
nothing # hide output

plot_seq(seq; show_adc=true, range=[2900, 3325], slider=true)

# ╔═╡ 1a62ae71-58db-49ea-ae6a-9aea66145963

phantom_T1 = plot(
	scatter(
		x=obj.x * 1e3,
		y=obj.T1 * 1e3,
		mode="markers",
		marker=attr(;
			color=obj.T1 * 1e3,
			colorscale=[
				[0.0, "black"],
				[183.0/maximum(obj.T1 .* 1e3), "green"],
				[750.0/maximum(obj.T1 .* 1e3), "blue"],
				[1122.0/maximum(obj.T1 .* 1e3), "red"],
			],
			cmin=0.0,
			cmax=1122.0,
			colorbar=attr(;ticksuffix="ms", title="T1"),
			showscale=false
		),
		showlegend=false
	)
)
relayout!(
	phantom_T1,
	yaxis_title="T1 [ms]",
	xaxis_title="x [mm]",
	xaxis_tickmode="array",
	xaxis_tickvals=[-1.5/2, 0.0, 1.5/2],
	yaxis_tickmode="array",
	yaxis_tickvals=unique(obj.T1 * 1e3),
	xaxis_range=[-1.5, 1.5],
	yaxis_range=[0.0, 1200.0],
	title="T1 map of 1D Phantom"
)
phantom_T2 = plot(
	scatter(
		x=obj.x * 1e3,
		y=obj.T2 * 1e3,
		mode="markers",
		marker=attr(;
			color=obj.T2 * 1e3,
			colorscale=[
				[0.0, "black"],
				[90.0/maximum(obj.T2 .* 1e3), "blue"],
				[93.0/maximum(obj.T2 .* 1e3), "green"],
				[263.0/maximum(obj.T2 .* 1e3), "red"],
			],
			cmin=0.0,
			cmax=263.0,
			colorbar=attr(;ticksuffix="ms", title="T2"),
			showscale=false
		),
		showlegend=false
	)
)
relayout!(
	phantom_T2,
	yaxis_title="T2 [ms]",
	xaxis_title="x [mm]",
	xaxis_tickmode="array",
	xaxis_tickvals=[-1.5/2, 0.0, 1.5/2],
	yaxis_tickmode="array",
	yaxis_tickvals=unique(obj.T2 * 1e3),
	xaxis_range=[-1.5, 1.5],
	yaxis_range=[0.0, 300.0],
	title="T2 map of 1D Phantom"
)
[phantom_T1 phantom_T2]

    # Prep plots
labs = ["Carotid", "Blood", "Fat"]
cols = ["blue", "red", "green"]
spin_group = [(1:Niso)', (Niso+1:2Niso)', (2Niso+1:3Niso)']
t = KomaMRICore.get_adc_sampling_times(seq)
Mxy(i) = abs.(sum(magnetization[:,spin_group[i],1,1][:,1,:],dims=2)[:]/length(spin_group[i]))
Mz(i) = real.(sum(magnetization[:,spin_group[i],2,1][:,1,:],dims=2)[:]/length(spin_group[i]))

# Plot
p0 = make_subplots(
	rows=2,
	cols=1,
	subplot_titles=["Mxy" "Mz" "Sequence"],
	shared_xaxes=true,
	vertical_spacing=0.1
)
for i=eachindex(spin_group)
	p1 = scatter(
		x=t, y=Mxy(i),
		name=labs[i],
		legendgroup=labs[i],
		marker_color=cols[i]
	)
	p2 = scatter(
		x=t,
		y=Mz(i),
		name=labs[i],
		legendgroup=labs[i],
		showlegend=false,
		marker_color=cols[i]
	)
	add_trace!(p0, p1, row=1, col=1)
	add_trace!(p0, p2, row=2, col=1)
end
relayout!(p0,
	yaxis_range=[0, 0.4],
	xaxis_range=[RR*dummy_heart_beats, RR*dummy_heart_beats+.250]
)
p0

FAs = 30:5:180 		# flip angle [deg]
RRs = 60 ./ (55:10:85)  # RR [s]
mag1 = zeros(ComplexF64, im_segments, Niso*3, length(FAs), length(RRs))
@progress for (m, RR) = enumerate(RRs), (n, im_flip_angle) = enumerate(FAs)
	seq_params1 = merge(seq_params, (; im_flip_angle, RR))
	sim_params1 = merge(sim_params, Dict("sim_method"=>BlochDict()))
	seq1        = CMRA(seq_params1...)
	obj1        = cardiac_phantom(0)
	magaux = @suppress simulate(obj1, seq1, sys; sim_params=sim_params1)
	mag1[:, :, n, m] .= magaux[end-im_segments+1:end, :] 
end

FFAs = 20:20:250 						 # flip angle [deg]
Δfs = (-1:0.2:1) .* (γ * sys.B0 * 1e-6)  # off-resonance Δf [s]
mag2 = zeros(ComplexF64, im_segments, Niso*3, length(FFAs), length(Δfs))
@progress for (m, Δf) = enumerate(Δfs), (n, FatSat_flip_angle) = enumerate(FFAs)
	seq_params2 = merge(seq_params, (; FatSat_flip_angle))
	sim_params2 = merge(sim_params, Dict("sim_method"=>BlochDict()))
	seq2        = CMRA(seq_params2...)
	obj2        = cardiac_phantom(Δf)
	magaux = @suppress simulate(obj2, seq2, sys; sim_params=sim_params2)
	mag2[:, :, n, m] .= magaux[end-im_segments+1:end, :] # Last heartbeat
end

T2ps = (20:5:80) # T2prep [ms]
mag3 = zeros(ComplexF64, im_segments, Niso*3, length(T2ps), length(RRs))
@progress for (m, RR) = enumerate(RRs), (n, T2p) = enumerate(T2ps)
	# Create the T2prep sequence for this specific duration
	seq_params3 = merge(seq_params, (; T2prep_duration=T2p*1e-3, RR))
	sim_params3 = merge(sim_params, Dict("sim_method"=>BlochDict()))
	seq3        = CMRA(seq_params3...)
	obj3        = cardiac_phantom(0)
	magaux = @suppress simulate(obj3, seq3, sys; sim_params=sim_params3)
	mag3[:, :, n, m] .= magaux[end-im_segments+1:end, :]
end

# Labels
labels = ["Carotid", "Blood", "Fat (T₁=183 ms)"]
colors = ["blue", "red", "green", "purple"]
spins = [(1:Niso)', ((Niso + 1):(2Niso))', ((2Niso + 1):(3Niso))']
mean(x, dim) = sum(x; dims=dim) / size(x, dim)
std(x, dim; mu=mean(x, dim)) = sqrt.(sum(abs.(x .- mu) .^ 2; dims=dim) / (size(x, dim) - 1))

# Reducing tissues's signal
signal_myoc = reshape(
	mean(abs.(mean(mag1[:, spins[1], :, :], 3)), 1), length(FAs), length(RRs)
)
signal_bloo = reshape(
	mean(abs.(mean(mag1[:, spins[2], :, :], 3)), 1), length(FAs), length(RRs)
)
diff_bloo_myoc = abs.(signal_bloo .- signal_myoc)
# Mean
mean_myoc = mean(signal_myoc, 2)
mean_bloo = mean(signal_bloo, 2)
mean_diff = mean(diff_bloo_myoc,2)
# Std
std_myoc  = std(signal_myoc, 2)
std_bloo  = std(signal_bloo, 2)
std_diff = std(diff_bloo_myoc,2)
# Plotting results
# Mean
s1 = scatter(;
	x=FAs,
	y=mean_myoc[:],
	name=labels[1],
	legendgroup=labels[1],
	line=attr(; color=colors[1]),
)
s2 = scatter(;
	x=FAs,
	y=mean_bloo[:],
	name=labels[2],
	legendgroup=labels[2],
	line=attr(; color=colors[2]),
)
s3 = scatter(;
	x=FAs,
	y=mean_diff[:],
	name="|Blood-Caro|",
	legendgroup="|Blood-Caro|",
	line=attr(color=colors[3])
)
# Std
s4 = scatter(;
	x=[FAs; reverse(FAs)],
	y=[(mean_myoc .- std_myoc)[:]; reverse((mean_myoc .+ std_myoc)[:])],
	name=labels[1],
	legendgroup=labels[1],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,0,255,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
s5 = scatter(;
	x=[FAs; reverse(FAs)],
	y=[(mean_bloo .- std_bloo)[:]; reverse((mean_bloo .+ std_bloo)[:])],
	name=labels[2],
	legendgroup=labels[2],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
s6 = scatter(;
	x=[FAs; reverse(FAs)],
	y=[(mean_diff .- std_diff)[:]; reverse((mean_diff .+ std_diff)[:])],
	name="|Blood-Caro|",legendgroup="|Blood-Caro|",
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,255,0.2)",
	line=attr(color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
# Plots
fig = plot([s1, s2, s3, s4, s5, s6])
relayout!(
	fig;
	yaxis=attr(; title="Signal [a.u.]", tickmode="array"),
	xaxis=attr(;
		title="Flip angle [deg]",
		tickmode="array",
		tickvals=[FAs[1], 85, 110, 130, FAs[end]],
		constrain="domain",
	),
	font=attr(; family="CMU Serif", size=16, scaleanchor="x", scaleratio=1),
	yaxis_range=[0, 0.3],
	xaxis_range=[FAs[1], FAs[end]],
	width=600,
	height=400,
	hovermode="x unified",
)
fig

## Calculating results
signal_myoc2 = reshape(
	mean(abs.(mean(mag2[:, spins[1], :, :], 3)), 1), length(FFAs), length(Δfs)
)
signal_bloo2 = reshape(
	mean(abs.(mean(mag2[:, spins[2], :, :], 3)), 1), length(FFAs), length(Δfs)
)
signal_fat2 = reshape(
	mean(abs.(mean(mag2[:, spins[3], :, :], 3)), 1), length(FFAs), length(Δfs)
)
#signal_fat22 = reshape(
	#mean(abs.(mean(mag2[:, spins[4], :, :], 3)), 1), length(FFAs), length(Δfs)
#)
mean_myoc2 = mean(signal_myoc2, 2)
mean_bloo2 = mean(signal_bloo2, 2)
mean_fat2  = mean(signal_fat2, 2)
#mean_fat22 = mean(signal_fat22, 2)
std_myoc2  = std(signal_myoc2, 2)
std_bloo2  = std(signal_bloo2, 2)
std_fat2   = std(signal_fat2, 2)
#std_fat22  = std(signal_fat22, 2)
# Plotting results
# Mean
s12 = scatter(;
	x=FFAs,
	y=mean_myoc2[:],
	name=labels[1],
	legendgroup=labels[1],
	line=attr(; color=colors[1]),
)
s22 = scatter(;
	x=FFAs,
	y=mean_bloo2[:],
	name=labels[2],
	legendgroup=labels[2],
	line=attr(; color=colors[2]),
)
s32 = scatter(;
	x=FFAs,
	y=mean_fat2[:],
	name=labels[3],
	legendgroup=labels[3],
	line=attr(; color=colors[3]),
)
# Std
s42 = scatter(;
	x=[FFAs; reverse(FFAs)],
	y=[(mean_myoc2 .- std_myoc2)[:]; reverse((mean_myoc2 .+ std_myoc2)[:])],
	name=labels[1],
	legendgroup=labels[1],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,0,255,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none",
)
s62 = scatter(;
	x=[FFAs; reverse(FFAs)],
	y=[(mean_bloo2 .- std_bloo2)[:]; reverse((mean_bloo2 .+ std_bloo2)[:])],
	name=labels[2],
	legendgroup=labels[2],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none",
)
s72 = scatter(;
	x=[FFAs; reverse(FFAs)],
	y=[(mean_fat2 .- std_fat2)[:]; reverse((mean_fat2 .+ std_fat2)[:])],
	name=labels[3],
	legendgroup=labels[3],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,255,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none",
)
# Plots
fig2 = plot([s12, s22, s32, s42, s62, s72])
relayout!(
	fig2;
	yaxis=attr(; title="Signal [a.u.]", tickmode="array"),
	xaxis=attr(;
		title="FatSat flip angle [deg]",
		tickmode="array",
		tickvals=[FFAs[1], 130, 150, 180, FFAs[end]],
		constrain="domain",
	),
	font=attr(; family="CMU Serif", size=16, scaleanchor="x", scaleratio=1),
	yaxis_range=[0, 0.4],
	xaxis_range=[FFAs[1], FFAs[end]],
	width=600,
	height=400,
	hovermode="x unified",
)
fig2

#T2prep results
signal_myoc3 = reshape(
	mean(abs.(mean(mag3[:, spins[1], :, :], 3)), 1), length(T2ps), length(RRs)
)
signal_bloo3 = reshape(
	mean(abs.(mean(mag3[:, spins[2], :, :], 3)), 1), length(T2ps), length(RRs)
)
diff_bloo_myoc3 = abs.(signal_bloo3 .- signal_myoc3)

mean_myoc3 = mean(signal_myoc3, 2)
mean_bloo3 = mean(signal_bloo3, 2)
mean_diff3 = mean(diff_bloo_myoc3, 2)

std_myoc3  = std(signal_myoc3, 2)
std_bloo3  = std(signal_bloo3, 2)
std_diff3 = std(diff_bloo_myoc3, 2)
# Plotting results
# Mean
s13 = scatter(;
	x=T2ps,
	y=mean_myoc3[:],
	name=labels[1],
	legendgroup=labels[1],
	line=attr(; color=colors[1]),
)
s23 = scatter(;
	x=T2ps,
	y=mean_bloo3[:],
	name=labels[2],
	legendgroup=labels[2],
	line=attr(; color=colors[2]),
)
s33 = scatter(;
	x=T2ps,
	y=mean_diff3[:],
	name="|Blood-Caro|",
	legendgroup="|Blood-Caro|",
	line=attr(color=colors[3])
)
# Std
s43 = scatter(;
	x=[T2ps; reverse(T2ps)],
	y=[(mean_myoc3 .- std_myoc3)[:]; reverse((mean_myoc3 .+ std_myoc3)[:])],
	name=labels[1],
	legendgroup=labels[1],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,0,255,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
s53 = scatter(;
	x=[T2ps; reverse(T2ps)],
	y=[(mean_bloo3 .- std_bloo3)[:]; reverse((mean_bloo3 .+ std_bloo3)[:])],
	name=labels[2],
	legendgroup=labels[2],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
s63 = scatter(;
	x=[T2ps; reverse(T2ps)],
	y=[(mean_diff3 .- std_diff3)[:]; reverse((mean_diff3 .+ std_diff3)[:])],
	name="|Blood-Caro|",legendgroup="|Blood-Caro|",
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,255,0.2)",
	line=attr(color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
# Plots
fig3 = plot([s13, s23, s33, s43, s53, s63])
relayout!(
	fig3;
	yaxis=attr(; title="Signal [a.u.]", tickmode="array"),
	xaxis=attr(;
		title="T2prep duration [ms]",
		tickmode="array",
		tickvals=[T2ps[1], 30, 50, 70, T2ps[end]],
		constrain="domain",
	),
	font=attr(; family="CMU Serif", size=16, scaleanchor="x", scaleratio=1),
	yaxis_range=[0, 0.3],
	xaxis_range=[T2ps[1], T2ps[end]],
	width=600,
	height=400,
	hovermode="x unified",
)
fig3
