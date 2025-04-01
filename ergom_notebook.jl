using Plots
include("ergom.jl")

# Initialize the ERGOM model
mod = ERGOM()

# Set initial conditions and parameters
mod.I_0 = 50.0  # Surface light intensity
mod.par = 10.0  # Photosynthetically active radiation
mod.temp = 15.0 # Temperature in degrees Celsius
mod.salt = 35.0 # Salinity in PSU
mod.wind = 5.0  # Wind speed in m/s
mod.taub = 0.05 # Bottom stress in N/m²

# Run the processes
process_interior(mod)
process_bottom(mod)
process_surface(mod)

# Collect results
state_variables = Dict(
    "Diatoms (p1)" => mod.p1,
    "Flagellates (p2)" => mod.p2,
    "Cyanobacteria (p3)" => mod.p3,
    "Zooplankton (zo)" => mod.zo,
    "Detritus (de)" => mod.de,
    "Oxygen (o2)" => mod.o2,
    "Ammonium (am)" => mod.am,
    "Nitrate (ni)" => mod.ni,
    "Phosphate (po)" => mod.po
)

# Plot results
plot_titles = ["State Variables"]
plot_colors = [:blue, :green, :red, :orange, :purple, :cyan, :magenta, :yellow, :gray]

plot()
for (i, (name, value)) in enumerate(state_variables)
    plot!(value, label=name, color=plot_colors[i])
end

title!("ERGOM Model State Variables")
xlabel!("Time Step")
ylabel!("Concentration")
legend()
savefig("ergom_results.png")

# Print final state variables
println("Final State Variables:")
for (name, value) in state_variables
    println("$name: $value")
end
