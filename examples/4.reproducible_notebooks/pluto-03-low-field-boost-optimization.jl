
# ╔═╡ 0b7a405e-bbb5-11ee-05ca-4b1c8567398d
using KomaMRICore, KomaMRIPlots, KomaMRIFiles, PlotlyJS # Essentials

# ╔═╡ 70dbc2bd-8b93-471d-8340-04d98a008ca6
using Suppressor, ProgressLogging, WebIO # Extras

# ╔═╡ 9e397426-b60b-4b98-be8b-f7f128621c44

sys = Scanner()
sys.B0 = 0.55
sys.Gmax = 40.0e-3
sys.Smax = 25.0
sys


# General sequence parameters
Trf = 500e-6  			# 500 [ms]
B1 = 1 / (360*γ*Trf)    # B1 amplitude [uT]
Tadc = 1e-6 			# 1us

# Prepulses
Tfatsat = 26.624e-3     # 26.6 [ms]
T2prep_duration = 50e-3 # 50 [ms]

# Acquisition
RR = 0.8 				# 1 [s]
dummy_heart_beats = 3 	# Steady-state
TR = 7.14e-3             # 5.3 [ms] RF Low SAR
TE = TR / 2 			# bSSFP condition
iNAV_lines = 3          # FatSat-Acq delay: iNAV_lines * TR (reduced from 3)
iNAV_flip_angle = 3.2   # 3.2 [deg]
im_segments = 30        # Acquisitino window: im_segments * TR (reduced from 30)

# To be optimized
im_flip_angle = [155, 90] # 80 [deg]
FatSat_flip_angle = 180   # 180 [deg]
IR_inversion_time = 90e-3 # 90 [ms] 
#t2p_50 = read_seq("examples/4.reproducible_notebooks/boost_055T_basico.seq")
#t2p_50 = read_seq("examples/4.reproducible_notebooks/adiabatic_t2_50ms.seq")
#t2p_50 = read_seq("examples/4.reproducible_notebooks/t2_mlev8_50.seq")

seq_params = (;
	dummy_heart_beats,
	iNAV_lines,
	im_segments,
	iNAV_flip_angle,
	im_flip_angle,
	T2prep_duration,
	#t2p_50,
	IR_inversion_time,
	FatSat_flip_angle,
	RR
)

seq_params

# ╔═╡ f0a81c9f-5616-4663-948f-a4084e1719af

fat_ppm = -3.4e-6 			# -3.4ppm fat-water frequency shift
Niso = 200        			# 200 isochromats in spoiler direction
Δx_voxel = 0.96e-3 			# 1.5 [mm]
fat_freq = γ*sys.B0*fat_ppm # -80 [Hz]
dx = Array(range(-Δx_voxel/2, Δx_voxel/2, Niso))


# ╔═╡ 6b870443-7be5-4287-b957-ca5c14eda89c

function FatSat(α, Δf; sample=false)
	# FatSat design
	# cutoff_freq = sqrt(log(2) / 2) / a where B1(t) = exp(-(π t / a)^2)
	cutoff = fat_freq / π 			      # cutoff [Hz] => ≈1/10 RF power to water
	a = sqrt(log(2) / 2) / cutoff         # a [s]
	τ = range(-Tfatsat/2, Tfatsat/2, 64)  # time [s]
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

function T2prep(TE; sample=false)
	seq = Sequence()
	seq += RF(90 * B1, Trf)
	seq += sample ? ADC(20, TE/2 - 1.5Trf) : Delay(TE/2 - 1.5Trf)
	seq += RF(180im * B1 / 2, Trf*2)
	seq += sample ? ADC(20, TE/2 - 1.5Trf) : Delay(TE/2 - 1.5Trf)
	seq += RF(-90 * B1, Trf)
	seq += Grad(8e-3, 6000e-6, 600e-6) #Spoiler3
	if sample
		seq += ADC(1, 1e-6)
	end
	return seq
end

function IR(IR_delay; sample=false)
	# Generating HS pulse
	# Based on: https://onlinelibrary.wiley.com/doi/epdf/10.1002/jmri.26021
	# Params
	flip_angle = 900;    # Peak amplitude (deg)
	Trf = 10240e-6;      # Pulse duration (ms)
	β = 6.7e2;           # frequency modulation param (rad/s)
	μ = 5;               # phase modulation parameter (dimensionless)
	fmax = μ * β / (2π); # 2fmax = BW
	# RF pulse
	t = range(-Trf/2, Trf/2, 201);
	B1 = sech.(β .* t);
	Δf = fmax  .* tanh.(β .* t);
	# Spoiler length
	spoiler_time = 6000e-6
	spoiler_rise_fall = 600e-6
	# Prepulse
	seq = Sequence()
	seq += RF(B1, Trf, Δf) # FM modulated pulse
	seq = (flip_angle / get_flip_angles(seq)[1] + 0.0im) * seq # RF scaling
	seq += Grad(8e-3, spoiler_time, spoiler_rise_fall) #Spoiler3
	if sample
		seq += ADC(11, IR_delay - spoiler_time - 2spoiler_rise_fall)
	else
		seq += Delay(IR_delay - spoiler_time - 2spoiler_rise_fall)
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
# ╔═╡ 7890f81e-cb15-48d2-a80c-9d73f9516056
function BOOST(
			dummy_heart_beats,
			iNAV_lines,
			im_segments,
			iNAV_flip_angle,
			im_flip_angle,
			T2prep_duration,
			#t2p_50,
			IR_inversion_time=70e-3,
			FatSat_flip_angle=180,
			RR=1.0;
			sample_recovery=zeros(Bool, dummy_heart_beats+1)
			)
	# Seq init
	seq = Sequence()
	for hb = 1 : dummy_heart_beats + 1
		sample = sample_recovery[hb] # Sampling recovery curve for hb
		# Generating seq blocks
		t2p = T2prep(T2prep_duration; sample)
		#t2p = t2p_50
		ir = IR(IR_inversion_time - iNAV_lines * TR - Trf - TE; sample)
		fatsat = FatSat(FatSat_flip_angle, fat_freq; sample)
		# Magnetization preparations
		for contrast = 1:2
			preps = Sequence()
			if contrast == 1 # Bright-blood contrast
				preps += t2p
				preps += ir
			else # Reference contrast
				preps += fatsat
			end
			# Contrst dependant flip angle
			bssfp = bSSFP(iNAV_lines, im_segments, iNAV_flip_angle,
				im_flip_angle[contrast]; sample)
			# Concatenating seq blocks
			seq += preps
			seq += bssfp
			# RR interval consideration
			RRdelay = RR  - dur(bssfp) - dur(preps)
			seq += sample ? ADC(80, RRdelay) : Delay(RRdelay)
		end
	end
	return seq
end

# ╔═╡ f57a2b6c-eb4c-45bd-8058-4a60b038925d

function cardiac_phantom(off; off_fat=fat_freq)
	myocard = Phantom(x=dx, ρ=0.6*ones(Niso), T1=750e-3*ones(Niso),
								T2=90e-3*ones(Niso),    Δw=2π*off*ones(Niso))
	blood =   Phantom(x=dx, ρ=0.7*ones(Niso), T1=1122e-3*ones(Niso),
								T2=263e-3*ones(Niso),   Δw=2π*off*ones(Niso))
	fat1 =    Phantom(x=dx, ρ=1.0*ones(Niso), T1=183e-3*ones(Niso),
								T2=93e-3*ones(Niso),    Δw=2π*(off_fat + off)*ones(Niso))
	fat2 =    Phantom(x=dx, ρ=1.0*ones(Niso), T1=130e-3*ones(Niso),
								T2=93e-3*ones(Niso),    Δw=2π*(off_fat + off)*ones(Niso))
	#obj = myocard + blood + fat1 + fat2
	obj = myocard + blood + fat1
	return obj
end

# ╔═╡ d05dcba7-2f42-47bf-a172-6123d0113b3f
sim_params = Dict{String,Any}(
	"return_type"=>"mat",
	"sim_method"=>BlochDict(save_Mz=true),
	"Δt_rf"=>Trf,
	"gpu"=>false,
	"Nthreads"=>1
)

# ╔═╡ 37f7fd7f-5cb1-48b5-b877-b2bc23a1e7dd

seq = BOOST(seq_params...; sample_recovery=ones(Bool, dummy_heart_beats+1))
obj = cardiac_phantom(0)
magnetization = @suppress simulate(obj, seq, sys; sim_params=sim_params)
nothing # hide output


# ╔═╡ 0b6c1f72-b040-483c-969b-88bfe09b32c3
plot_seq(seq; range=[5990, 6280], slider=true)

# ╔═╡ 88eb41a5-d8c2-4f0e-b379-a8b05a341a82
plot_seq(seq; range=[6900, 7190], slider=true)

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
				[450.0/maximum(obj.T1 .* 1e3), "blue"],
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
				[54.0/maximum(obj.T2 .* 1e3), "blue"],
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

# ╔═╡ d9715bc1-49cd-4df8-8dbf-c06de42ad550
    # Prep plots
labs = ["Muscle", "Blood", "Fat"]
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

FAs = 20:5:180          # flip angle [deg]
RRs = 60 ./ (55:10:85)  # RR [s]
mag1 = zeros(ComplexF64, im_segments, Niso*3, length(FAs), length(RRs))
@progress for (m, RR) = enumerate(RRs), (n, α) = enumerate(FAs)
	seq_params1 = merge(seq_params, (; im_flip_angle=[110, α], RR))
	sim_params1 = merge(sim_params, Dict("sim_method"=>BlochDict()))
	seq1        = BOOST(seq_params1...)
	obj1        = cardiac_phantom(0)
	magaux = @suppress simulate(obj1, seq1, sys; sim_params=sim_params1)
	mag1[:, :, n, m] .= magaux[end-im_segments+1:end, :] # Last heartbeat
end

FFAs = 20:20:250 						 # flip angle [deg]
Δfs = (-1:0.2:1) .* (γ * sys.B0 * 1e-6)  # off-resonance Δf [s]
mag2 = zeros(ComplexF64, im_segments, Niso*3, length(FFAs), length(Δfs))
@progress for (m, Δf) = enumerate(Δfs), (n, FatSat_flip_angle) = enumerate(FFAs)
	seq_params2 = merge(seq_params, (; FatSat_flip_angle))
	sim_params2 = merge(sim_params, Dict("sim_method"=>BlochDict()))
	seq2        = BOOST(seq_params2...)
	obj2        = cardiac_phantom(Δf)
	magaux = @suppress simulate(obj2, seq2, sys; sim_params=sim_params2)
	mag2[:, :, n, m] .= magaux[end-im_segments+1:end, :] # Last heartbeat
end

# ╔═╡ 14cf6859-a4b9-4671-b022-659781c55144

mag4 = zeros(ComplexF64, im_segments, Niso*3, length(FAs), length(RRs))
@progress for (m, RR) = enumerate(RRs), (n, α) = enumerate(FAs)
	seq_params4 = merge(seq_params, (; im_flip_angle=[α, 80], RR))
	sim_params4 = merge(sim_params, Dict("sim_method"=>BlochDict()))
	seq4        = BOOST(seq_params4...)
	obj4        = cardiac_phantom(0)
	magaux = @suppress simulate(obj4, seq4, sys; sim_params=sim_params4)
	mag4[:, :, n, m] .= magaux[end-2im_segments+1:end-im_segments, :] # Bright-Blood
end

# ╔═╡ c952ecf1-25ef-4b48-9c8f-e53ded302629

TIs = (40:5:100) # Inversion delay [ms]
mag3 = zeros(ComplexF64, im_segments, Niso*3, length(TIs), length(Δfs))
@progress for (m, Δf) = enumerate(Δfs), (n, TI) = enumerate(TIs)
	seq_params3 = merge(seq_params, (; IR_inversion_time=TI * 1e-3, RR))
	sim_params3 = merge(sim_params, Dict("sim_method"=>BlochDict()))
	seq3        = BOOST(seq_params3...)
	obj3        = cardiac_phantom(Δf)
	magaux = @suppress simulate(obj3, seq3, sys; sim_params=sim_params3)
	mag3[:, :, n, m] .= magaux[end-2im_segments+1:end-im_segments, :] # Bright-Blood
end


# ╔═╡ 85834365-238b-4193-a0c0-d1859416ab6c

T2ps = (20:5:80) # T2prep duration [ms]
mag5bb = zeros(ComplexF64, im_segments, Niso*3, length(T2ps), length(RRs))
mag5rf = zeros(ComplexF64, im_segments, Niso*3, length(T2ps), length(RRs))
@progress for (m, RR) = enumerate(RRs), (n, T2p) = enumerate(T2ps)
	seq_params5 = merge(seq_params, (; T2prep_duration=T2p * 1e-3, RR))
	sim_params5 = merge(sim_params, Dict("sim_method"=>BlochDict()))
	seq5        = BOOST(seq_params5...)
	obj5        = cardiac_phantom(0)
	magaux = @suppress simulate(obj5, seq5, sys; sim_params=sim_params5)
	# Bright-Blood
	mag5bb[:, :, n, m] .= magaux[end-2im_segments+1:end-im_segments, :]
	# Reference contrast
	mag5rf[:, :, n, m] .= magaux[end-im_segments+1:end, :] 
end

# Labels
labels = ["Muscle", "Blood", "Fat (T₁=183 ms)"]
colors = ["blue", "red", "purple"]
#spins = [(1:Niso)', ((Niso + 1):(2Niso))', ((2Niso + 1):(3Niso))', ((3Niso + 1):(4Niso))']
spins = [(1:Niso)', ((Niso + 1):(2Niso))', ((2Niso + 1):(3Niso))']
mean(x, dim) = sum(x; dims=dim) / size(x, dim)
std(x, dim; mu=mean(x, dim)) = sqrt.(sum(abs.(x .- mu) .^ 2; dims=dim) / (size(x, dim) - 1))


# ╔═╡ f73082ff-a6d3-41f8-8796-4114fa89d2bb
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
	name="|Blood-Muscle|",
	legendgroup="|Blood-Muscle|",
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
	name="|Blood-Muscle|",legendgroup="|Blood-Muscle|",
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
		tickvals=[FAs[1], 80, 90, 110, 130, FAs[end]],
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

# ╔═╡ 44a31057-7b34-4c80-a273-6621c0773dc7
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
#= signal_fat22 = reshape(
	mean(abs.(mean(mag2[:, spins[4], :, :], 3)), 1), length(FFAs), length(Δfs)
) =#
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
#= s52 = scatter(;
	x=FFAs,
	y=mean_fat22[:],
	name=labels[4],
	legendgroup=labels[4],
	line=attr(; color=colors[3], dash="dash"),
) =#
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
#= s82 = scatter(;
	x=[FFAs; reverse(FFAs)],
	y=[(mean_fat22 .- std_fat22)[:]; reverse((mean_fat22 .+ std_fat22)[:])],
	name=labels[4],
	legendgroup=labels[4],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,255,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none",
) =#
# Plots
#fig2 = plot([s12, s22, s32, s42, s52, s62, s72, s82])
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

# ╔═╡ 6649f46e-3565-4cd6-86a5-9b03d00cc3db
# Reducing tissues's signal
signal_myoc4 = reshape(
	mean(abs.(mean(mag4[:, spins[1], :, :], 3)), 1), length(FAs), length(RRs)
)
signal_bloo4 = reshape(
	mean(abs.(mean(mag4[:, spins[2], :, :], 3)), 1), length(FAs), length(RRs)
)
diff_bloo_myoc4 = abs.(signal_bloo4 .- signal_myoc4)
# Mean
mean_myoc4 = mean(signal_myoc4, 2)
mean_bloo4 = mean(signal_bloo4, 2)
mean_diff4 = mean(diff_bloo_myoc4,2)
# Std
std_myoc4  = std(signal_myoc4, 2)
std_bloo4  = std(signal_bloo4, 2)
std_diff4 = std(diff_bloo_myoc4,2)
# Plotting results
# Mean
s14 = scatter(;
	x=FAs,
	y=mean_myoc4[:],
	name=labels[1],
	legendgroup=labels[1],
	line=attr(; color=colors[1]),
)
s24 = scatter(;
	x=FAs,
	y=mean_bloo4[:],
	name=labels[2],
	legendgroup=labels[2],
	line=attr(; color=colors[2]),
)
s34 = scatter(;
	x=FAs,
	y=mean_diff4[:],
	name="|Blood-Muscle|",
	legendgroup="|Blood-Muscle|",
	line=attr(color=colors[3])
)
# Std
s44 = scatter(;
	x=[FAs; reverse(FAs)],
	y=[(mean_myoc4 .- std_myoc4)[:]; reverse((mean_myoc4 .+ std_myoc4)[:])],
	name=labels[1],
	legendgroup=labels[1],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,0,255,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
s54 = scatter(;
	x=[FAs; reverse(FAs)],
	y=[(mean_bloo4 .- std_bloo4)[:]; reverse((mean_bloo4 .+ std_bloo4)[:])],
	name=labels[2],
	legendgroup=labels[2],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
s64 = scatter(;
	x=[FAs; reverse(FAs)],
	y=[(mean_diff4 .- std_diff4)[:]; reverse((mean_diff4 .+ std_diff4)[:])],
	name="|Blood-Muscle|",legendgroup="|Blood-Muscle|",
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,255,0.2)",
	line=attr(color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
# Plots
fig4 = plot([s14, s24, s34, s44, s54, s64])
relayout!(
	fig4;
	yaxis=attr(; title="Signal [a.u.]", tickmode="array"),
	xaxis=attr(;
		title="Flip angle [deg]",
		tickmode="array",
		tickvals=[FAs[1], 110, FAs[end]],
		constrain="domain",
	),
	font=attr(; family="CMU Serif", size=16, scaleanchor="x", scaleratio=1),
	yaxis_range=[0, 0.3],
	xaxis_range=[FAs[1], FAs[end]],
	width=600,
	height=400,
	hovermode="x unified",
)
fig4

# ╔═╡ a8751a97-c131-407b-b211-573660452142
## Calculating results
signal_myoc3 = reshape(
	mean(abs.(mean(mag3[:, spins[1], :, :], 3)), 1), length(TIs), length(Δfs)
)
signal_bloo3 = reshape(
	mean(abs.(mean(mag3[:, spins[2], :, :], 3)), 1), length(TIs), length(Δfs)
)
signal_fat3 = reshape(
	mean(abs.(mean(mag3[:, spins[3], :, :], 3)), 1), length(TIs), length(Δfs)
)
#= signal_fat33 = reshape(
	mean(abs.(mean(mag3[:, spins[4], :, :], 3)), 1), length(TIs), length(Δfs)
) =#
mean_myoc3 = mean(signal_myoc3, 2)
mean_bloo3 = mean(signal_bloo3, 2)
mean_fat3  = mean(signal_fat3, 2)
#mean_fat33 = mean(signal_fat33, 2)
std_myoc3  = std(signal_myoc3, 2)
std_bloo3  = std(signal_bloo3, 2)
std_fat3   = std(signal_fat3, 2)
#std_fat33  = std(signal_fat33, 2)
# Plotting results
# Mean
s13 = scatter(;
	x=TIs,
	y=mean_myoc3[:],
	name=labels[1],
	legendgroup=labels[1],
	line=attr(; color=colors[1]),
)
s23 = scatter(;
	x=TIs,
	y=mean_bloo3[:],
	name=labels[2],
	legendgroup=labels[2],
	line=attr(; color=colors[2]),
)
s33 = scatter(;
	x=TIs,
	y=mean_fat3[:],
	name=labels[3],
	legendgroup=labels[3],
	line=attr(; color=colors[3]),
)
#= s53 = scatter(;
	x=TIs,
	y=mean_fat33[:],
	name=labels[4],
	legendgroup=labels[4],
	line=attr(; color=colors[3], dash="dash"),
) =#
# Std
s43 = scatter(;
	x=[TIs; reverse(TIs)],
	y=[(mean_myoc3 .- std_myoc3)[:]; reverse((mean_myoc3 .+ std_myoc3)[:])],
	name=labels[1],
	legendgroup=labels[1],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,0,255,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none",
)
s63 = scatter(;
	x=[TIs; reverse(TIs)],
	y=[(mean_bloo3 .- std_bloo3)[:]; reverse((mean_bloo3 .+ std_bloo3)[:])],
	name=labels[2],
	legendgroup=labels[2],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none",
)
s73 = scatter(;
	x=[TIs; reverse(TIs)],
	y=[(mean_fat3 .- std_fat3)[:]; reverse((mean_fat3 .+ std_fat3)[:])],
	name=labels[3],
	legendgroup=labels[3],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,255,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none",
)
#= s83 = scatter(;
	x=[TIs; reverse(TIs)],
	y=[(mean_fat33 .- std_fat33)[:]; reverse((mean_fat33 .+ std_fat33)[:])],
	name=labels[4],
	legendgroup=labels[4],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,255,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none",
) =#
# Plots
fig3 = plot([s13, s23, s33, s43, s63, s73])
relayout!(
	fig3;
	yaxis=attr(; title="Signal [a.u.]", tickmode="array"),
	xaxis=attr(;
		title="Inversion Time [ms]",
		tickmode="array",
		tickvals=[TIs[1], 70, 90, TIs[end]],
		constrain="domain",
	),
	font=attr(; family="CMU Serif", size=16, scaleanchor="x", scaleratio=1),
	yaxis_range=[0, 0.3],
	xaxis_range=[TIs[1], TIs[end]],
	width=600,
	height=400,
	hovermode="x unified",
)
fig3


# ╔═╡ eb0f1e07-24a1-4aa4-b3ef-b63caa97de34
# Reducing tissues's signal
signal_myoc5 = reshape(
	mean(abs.(mean(mag5bb[:, spins[1], :, :], 3)), 1), length(T2ps), length(RRs)
)
signal_bloo5 = reshape(
	mean(abs.(mean(mag5bb[:, spins[2], :, :], 3)), 1), length(T2ps), length(RRs)
)
diff_bloo_myoc5 = abs.(signal_bloo5 .- signal_myoc5)
# Mean
mean_myoc5 = mean(signal_myoc5, 2)
mean_bloo5 = mean(signal_bloo5, 2)
mean_diff5 = mean(diff_bloo_myoc5,2)
# Std
std_myoc5  = std(signal_myoc5, 2)
std_bloo5  = std(signal_bloo5, 2)
std_diff5 = std(diff_bloo_myoc5,2)
# Plotting results
# Mean
s15 = scatter(;
	x=T2ps,
	y=mean_myoc5[:],
	name=labels[1],
	legendgroup=labels[1],
	line=attr(; color=colors[1]),
)
s25 = scatter(;
	x=T2ps,
	y=mean_bloo5[:],
	name=labels[2],
	legendgroup=labels[2],
	line=attr(; color=colors[2]),
)
s35 = scatter(;
	x=T2ps,
	y=mean_diff5[:],
	name="|Blood-Muscle|",
	legendgroup="|Blood-Muscle|",
	line=attr(color=colors[3])
)
# Std
s45 = scatter(;
	x=[T2ps; reverse(T2ps)],
	y=[(mean_myoc5 .- std_myoc5)[:]; reverse((mean_myoc5 .+ std_myoc5)[:])],
	name=labels[1],
	legendgroup=labels[1],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,0,255,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
s55 = scatter(;
	x=[T2ps; reverse(T2ps)],
	y=[(mean_bloo5 .- std_bloo5)[:]; reverse((mean_bloo5 .+ std_bloo5)[:])],
	name=labels[2],
	legendgroup=labels[2],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
s65 = scatter(;
	x=[T2ps; reverse(T2ps)],
	y=[(mean_diff5 .- std_diff5)[:]; reverse((mean_diff5 .+ std_diff5)[:])],
	name="|Blood-Carotid|",legendgroup="|Blood-Carotid|",
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,255,0.2)",
	line=attr(color="rgba(0,0,0,0)"),
	hoverinfo="none"
)
# Plots
fig5 = plot([s15, s25, s35, s45, s55, s65])
relayout!(
	fig5;
	yaxis=attr(; title="Signal [a.u.]", tickmode="array"),
	xaxis=attr(;
		title="T2prep duration [ms]",
		tickmode="array",
		tickvals=[T2ps[1], 50, 70, T2ps[end]],
		constrain="domain",
	),
	font=attr(; family="CMU Serif", size=16, scaleanchor="x", scaleratio=1),
	yaxis_range=[0, 0.2],
	xaxis_range=[T2ps[1], T2ps[end]],
	width=600,
	height=400,
	hovermode="x unified",
)
fig5

# ╔═╡ 10e68e25-9f68-46c0-b1b0-e9b124706b67
# Reducing tissues's signal
#signal_myocbb = reshape(
	mean(abs.(mean(mag5bb[:, spins[1], :, :], 3)), 1), length(T2ps), length(RRs)
#)
#signal_bloodbb = reshape(
	mean(abs.(mean(mag5bb[:, spins[2], :, :], 3)), 1), length(T2ps), length(RRs)
#)
#signal_myocrf = reshape(
	mean(abs.(mean(mag5rf[:, spins[1], :, :], 3)), 1), length(T2ps), length(RRs)
#)
#signal_bloodrf = reshape(
	mean(abs.(mean(mag5rf[:, spins[2], :, :], 3)), 1), length(T2ps), length(RRs)
#)
# Substracted
#signal_myoc6 = abs.(signal_myocrf .- signal_myocbb)
#signal_bloo6 = abs.(signal_bloodrf .- signal_bloodbb)
#diff_bloo_myoc6 = abs.(signal_bloo6 .- signal_myoc6)
# Mean
#mean_myoc6 = mean(signal_myoc6, 2)
#mean_bloo6 = mean(signal_bloo6, 2)
#mean_diff6 = mean(diff_bloo_myoc6,2)
# Std
#std_myoc6  = std(signal_myoc6, 2)
#std_bloo6  = std(signal_bloo6, 2)
#std_diff6 = std(diff_bloo_myoc6,2)
# Plotting results
# Mean
#s16 = scatter(;
	x=T2ps,
	y=mean_myoc6[:],
	name="Substracted "*labels[1],
	legendgroup=labels[1],
	line=attr(; color=colors[1]),
#)
#s26 = scatter(;
	x=T2ps,
	y=mean_bloo6[:],
	name="Substracted "*labels[2],
	legendgroup=labels[2],
	line=attr(; color=colors[2]),
#)
#s36 = scatter(;
	x=T2ps,
	y=mean_diff6[:],
	name="|Sub. Blood - Sub. Muscle|",
	legendgroup="|Blood-Muscle|",
	line=attr(color=colors[3])
#)
# Std
#s46 = scatter(;
	x=[T2ps; reverse(T2ps)],
	y=[(mean_myoc6 .- std_myoc6)[:]; reverse((mean_myoc6 .+ std_myoc6)[:])],
	name="Substracted "*labels[1],
	legendgroup=labels[1],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(0,0,255,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none"
#)
#s56 = scatter(;
	x=[T2ps; reverse(T2ps)],
	y=[(mean_bloo6 .- std_bloo6)[:]; reverse((mean_bloo6 .+ std_bloo6)[:])],
	name="Substracted "*labels[2],
	legendgroup=labels[2],
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,0,0.2)",
	line=attr(; color="rgba(0,0,0,0)"),
	hoverinfo="none"
#)
#s66 = scatter(;
	x=[T2ps; reverse(T2ps)],
	y=[(mean_diff6 .- std_diff6)[:]; reverse((mean_diff6 .+ std_diff6)[:])],
	name="|Sub. Blood - Sub. Muscle|",legendgroup="|Blood-Muscle|",
	showlegend=false,
	fill="toself",
	fillcolor="rgba(255,0,255,0.2)",
	line=attr(color="rgba(0,0,0,0)"),
	hoverinfo="none"
#)
# Plots
#fig6 = plot([s16, s26, s36, s46, s56, s66])
#relayout!(
	#fig6;
	#yaxis=attr(; title="Signal [a.u.]", tickmode="array"),
	#xaxis=attr(;
		#title="T2prep duration [deg]",
		#tickmode="array",
		#tickvals=[T2ps[1], 50, 70, T2ps[end]],
		#constrain="domain",
	#),
	#font=attr(; family="CMU Serif", size=16, scaleanchor="x", scaleratio=1),
	#yaxis_range=[0, 0.2],
	#xaxis_range=[T2ps[1], T2ps[end]],
	#width=600,
	#height=400,
	#hovermode="x unified",
#)
#fig6
