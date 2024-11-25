//################################################################################
//# ROTOR CLASS
//################################################################################
/*
`Rotor(CW, r, chord, theta, LE_x, LE_z, B, airfoil)`

Object defining the geometry of a rotor / propeller / wind turbine.This class
behaves as an extension of the WingSystem class, hence all functions of
WingSystem can be applied to a Rotor object.

# Arguments
* CW::Bool                   : True for clockwise rotation, false for CCW.
* r::Array{ Float64,1 } : Radius position for the following variables.
* chord::Array{ Float64,1 } : Chord length.
* theta::Array{ Float64,1 } : Angle of attack(deg) from the rotor's plane
of rotation.
* LE_x::Array{ Float64,1 } : x - position of leading edge(positive is ahead
    of radial axis relative to rotation).
    * LE_z::Array{ Float64,1 } : z - position of leading edge(height from plane
        of rotation).
    * B::Int64 : Number of blades.

    # Optional Arguments
    * airfoils::Array{ Tuple{Float64, airfoilprep.Polar},1 } : 2D airfoil properties
    along blade in the form[(r_i, Polar_i)]
    with Polar_i describes the airfoil at i - th
    radial position r_i(both the airfoil geometry
        in Polar_i and r_i must be normalized).At
    least root(r = 0) and tip(r = 1) must be given
    so all positions in between can be
    extrapolated.This properties are only used
    when calling CCBlade and for generating good
    loking visuals; ignore if only solving the VLM.

    NOTE: r here is the radial position after precone is included in the geometry,
    hence the need of explicitely declaring LE_z.

    # PROPERTIES
    * `sol` : Contains solution fields specific for Rotor types.They are formated
    as sol[field_name] = Dict(
        "field_name" = > output_field_name,
        "field_type" = > "scalar" or "vector",
        "field_data" = > data
    )
    where `data` is an array data[i] = [val1, val2, ...] containing
this field values(scalar or vector) of all control points in the
i - th blade.

< !--NOTE TO SELF : r is the y - direction on a wing, hence, remember to build the
    blade from root in the direction of positive y. -->
    */