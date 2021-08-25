"""
STATE POINT ANALYSIS
by Tommy Thompson

Background
Wastewater treatment is a biochemical and mechanical process in which municipal and commercial wastewater
undergoes multiple, consecutive stages of separation to remove suspended and dissolved solids before
discharging treated effluent into a natural waterway.  The process train typically consists of:
     - Racks and screens to capture trash and other floatables
     - Primary sedimentation to settle out heavier suspended sediment or waste solids (sludge)
     - Aeration to promote bacterial growth for consuming remaining solids
     - Secondary sedimentation (clarification) to settle out waste-bacteria aggregates
     - Discharge of treated effluent into a local waterbody for finishing by natural microorganisms

In most setups, the aeration-secondary treatment stage occur simultaneously in a continuous-flow loop
process.  That is, part of the sludge blanket that settles out at the bottom of the secondary clarifier
is intentionally pumped back in to the aeration tank to more efficiently promote the growth of the
necessary microorganisms.  This is known as "Return Activated Sludge (RAS)," referring to it being pumped
back into the previous stage (return) and primed full of beneficial microorganisms (activated).

The volumetric flow rate at which the RAS is pumped into the aeration tank is controlled by the
Wastewater Operator.  The ultimate goal is to maintain a stable yet ever-present sludge blanket at the
bottom of the secondary clarifier: pump too much and it disappears; pump too little and it accumulates
and eventually leaves the clarifier as effluent.

To more easily and efficiently maintain this preferred operating point, an idealized model for secondary
treatment was created known as "State Point Analysis."  Given parameters such as the solids concentration
of the aeration tank (Mixed Liquor Suspended Solids) and the sludge volume index (a measure of how readily
the solids settle in water), the reactor hydraulics of the clarifier can be plotted to visualize whether
the operator is running the RAS pump too low or too high.

Description
The following module aims to maximize energy efficiency, minimize risk of losing solids, and simplify the
operation of secondary treatment by analytically determining the precise point below which the sludge
blanket would begin to rise and threaten to exit the system.

Future Work
Ultimately, the goal would be to integrate operational data into this tool as the inputs, such that the
output graph and Q_RAS values update during set time intervals to inform operators of what the most
efficient pump rate should be.

Date Last Modified - 08/17/2021
"""

# Required Libraries #
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from mpmath import lambertw

# Constants #
NUMBER_OF_TANKS = 3  # Number of clarifiers
TANK_AREA = 21382  # ft2; Area of each clarifier (length x width)

# Inputs #
test = "SVISN"  # Test used to determine sludge volume index
SVI = 140  # mL/g; Sludge Volume Index as determined by "test"
MLSS = 4.3  # g/L; Mixed Liquor Suspended Solids concentration
Qin = 36  # mgd; Influent flow


##############


# Functions #
def convert(val, orig_units):
    """
    This is a convenience function for converting units of common products or inputs into like units for later use.

    orig_units: m-g/h-L, lb/MG, MG-g/d-ft2-L, mgd
    """

    # Raise error if incorrect units entered.
    units_options = ["m-g/h-L", "lb/MG", "MG-g/d-ft2-L", "mgd"]
    if orig_units not in units_options:
        raise ValueError("Invalid units.  Expected one of {}".format(units_options))

    # Converts to lb/ft2-d (mass flow rate per unit area)
    if orig_units == units_options[0]:
        conv = 1 / 1000 * 100 ** 3 / 3.28084 ** 2 * 24 / 1000 * 2.20462
    # Converts to g/L (concentration)
    elif orig_units == units_options[1]:
        conv = 1 / 1000000 * 7.48 * 3.28084 ** 3 / 100 ** 3 * 1000 * 1000 / 2.20462
    # Converts to lb/ft2-d (mass flow rate per unit area)
    elif orig_units == units_options[2]:
        conv = 1000000 / 1000 * 1000000 / 3.28084 ** 3 / 7.48 * 2.20462 / 1000
    # Converts to lb-L/d-g (mass flow rate normalized by concentration)
    elif orig_units == units_options[3]:
        conv = 1 / convert(1, "lb/MG")

    return val * conv


def def_constants(test_type="SVISN"):
    """
    The State Point Analysis is based on modeled sludge settling characteristics.  The model relies on four constants
    determined empirically, with each one varying depending on the type of bench-scale test used to determine the sludge
    volume index.  While the generally preferred method is SVISN, the options include:

    SVISN: settleometer without stirring
    SVISS: settleometer with stirring
    SVIGN: graduated cylinder without stirring
    SVIGS: graduated cylinder with stirring

    test_type: SVISN, SVISS, SVIGN, SVIGS
    """

    # Test: (alpha (L/g), beta (L/mL), delta (g/mL), gamma (m/h))
    test_dict = {
        "SVISN": (0.261, 0.00170, 0.00370, 14.9),
        "SVISS": (0.211, 0.00236, 0.00593, 14.6),
        "SVIGN": (0.351, 0.00058, 0.00602, 18.2),
        "SVIGS": (0.245, 0.00296, 0.01073, 24.3)
    }

    constants = test_dict[test_type]

    return constants


def surface_overflow_calc(q_in, tank_area, number_of_tanks, sludge_volume_index):
    """
    The surface overflow rate (SOR) of a sedimentation tank (secondary clarifier) is representative of the module's
    performance in removing solids.  Obtained by dividing the flow rate by the surface area (width x length) of the
    tank, SOR can also be viewed as the settling velocity of the smallest particle the tank will remove.

    This function receives the influent flow rate, tank surface area, number of tanks in operation, and SVI (settling
    characteristics of the sludge) and returns two tuples: the x- and y-coordinates of the SOR line to be plotted on the
    SPA graph.
    """

    # SOR line begins at origin
    x_sor_1, y_sor_1 = 0, 0

    #       1.25*γ*e^(-(δ*SVI + 1))
    # Y_2 = -----------------------
    #             α + β*SVI
    y_sor_2 = 1.25 * gamma * np.exp(-(delta * sludge_volume_index + 1)) / (alpha + beta * sludge_volume_index)
    y_sor_2 = convert(y_sor_2, "m-g/h-L")

    #       Y_2*N*A
    # X_2 = -------
    #        Q_in
    x_sor_2 = y_sor_2 * number_of_tanks * tank_area / q_in
    x_sor_2 = convert(x_sor_2, "lb/MG")

    x = (x_sor_1, x_sor_2)
    y = (y_sor_1, y_sor_2)

    return x, y


def solids_underflow_calc(q_in, q_ras, mixed_liquor_ss, tank_area, number_of_tanks):
    """
    The solids underflow rate (SUR) refers to the downward mass flow rate of solids (sludge) as they settle in the
    clarifier.  The y-intercept (solids concentration of zero) represents the solids loading rate (SLR), the mass flow
    rate of influent solids normalized by clarifier surface area.  The x-intercept (solids flux of zero) represents the
    modeled return activated sludge (RAS), the concentration of sludge at the bottom of the clarifier (where settling
    has stopped).

    This function receives the influent flow rate, RAS flow rate, MLSS concentration, tank surface area, and number of
    tanks in operation and returns two tuples: the x- and y-coordinates of the SUR line to be plotted on the
    SPA graph.
    """

    # Assign for x- and y-intercepts
    x_sur_1, y_sur_2 = 0, 0

    #       (Q_in + Q_ras)*MLSS
    # Y_1 = -------------------
    #               N*A
    y_sur_1 = (q_in + q_ras) * mixed_liquor_ss / (number_of_tanks * tank_area)
    y_sur_1 = convert(y_sur_1, "MG-g/d-ft2-L")

    #       Y_1*N*A
    # X_2 = -------
    #        Q_ras
    x_sur_2 = y_sur_1 * number_of_tanks * tank_area / q_ras
    x_sur_2 = convert(x_sur_2, "lb/MG")

    x = (x_sur_1, x_sur_2)
    y = (y_sur_1, y_sur_2)

    return x, y


def flux(solids_conc, sludge_volume_index):
    """
    This function computes the y-value (solids flux) for a given solids concentration along the settling flux curve
    generated from the determination of SVI from empirical settling tests.
    """

    # Y = γ*C*e^(-(δ*SVI + (α + β*SVI)*C))
    y = gamma * solids_conc * np.exp(-(delta * sludge_volume_index
                                       + (alpha + beta * sludge_volume_index) * solids_conc))
    y = convert(y, "m-g/h-L")

    return y


def settle_flux(sludge_volume_index, solids_max=15):
    """
    To plot the settling flux curve, a range of solids concentrations (n = 100) with a given maximum concentration
    (default of 15 g/L) is passed to the previously defined flux() function to generate two arrays of coordinates
    corresponding to the x and y values.
    """

    # Generate equal-interval range of 100 solids concentrations from 0 to solids_max (15)
    x = np.linspace(0, solids_max, num=100)

    # Compute solids flux on settling flux curve for each solids concentration
    y = flux(x, sludge_volume_index)

    return x, y


def plot_spa(q_in, q_ras, tank_area, number_of_tanks, sludge_volume_index, mixed_liquor_ss, solids_max=15):
    """
    This function aims to plot the three curves characteristic of a State Point Analysis: the surface overflow rate
    (positive slope), the solids underflow rate (negative slope), and the settling flux curve, which models the settling
    behavior of the sludge as determined by its SVI.  The state point (point of operation) itself is also plotted.
    """

    # Acquire both x-y pairs/array for SOR, SUR, and settling flux curve
    x_overflow_rate, y_overflow_rate = surface_overflow_calc(q_in, tank_area, number_of_tanks, sludge_volume_index)
    x_underflow_rate, y_underflow_rate = solids_underflow_calc(q_in, q_ras, mixed_liquor_ss, tank_area, number_of_tanks)
    x_flux, y_flux = settle_flux(sludge_volume_index, solids_max)

    # Calculate SOR (slope) from SOR line endpoint coordinates
    surface_overflow_rate = y_overflow_rate[1] / x_overflow_rate[1]

    # Calculate state point (operating point) solids flux value (y) from MLSS (operating solids concentration, x) and
    # slope of SOR line
    state_point = surface_overflow_rate * mixed_liquor_ss

    # Plot SOR line, SUR line, and settling flux curve as well as state point
    plt.plot(x_overflow_rate, y_overflow_rate, "g", x_underflow_rate, y_underflow_rate, "orange", x_flux, y_flux, "b")
    plt.plot(mixed_liquor_ss, state_point, "ro")
    plt.axis([0, solids_max, 0, 60])
    plt.grid(True)

    plt.title("State Point Analysis")
    plt.xlabel("Solids Concentration (g/L)")
    plt.ylabel("Solids Flux (lb/ft2-d)")

    ras_sub = r'$Q_{RAS}$'
    txt = "%s = %s MGD" % (ras_sub, q_ras)
    plt.text(0.5 * solids_max, 2 * gamma, txt, size=16, backgroundcolor="white", bbox=dict(facecolor="white"))

    plt.show()


def oper_round(ref, val):
    """
    This function rounds a given value (Q_ras) to increasingly general multiples based on a reference (Q_in) value.

    oper_round(9, 5.7381) --> 5.8
    oper_round(57, 23.128) --> 23.25
    """

    if ref < 10:
        precision = 0.10
    elif ref < 100:
        precision = 0.25
    else:
        precision = 0.50

    op_qras = np.ceil(val / precision) * precision

    return op_qras


def find_state_point(sludge_volume_index, mixed_liquor_ss, q_in):
    """
    This function analytically calculates the most efficient operational state point such that the entirety of the
    SUR line remains under the settling flux curve to maintain a stable sludge blanket while also maximizing energy
    efficiency.

    Starting by determining the peak of the settling flux curve, a linear approximation of the falling tail of the
    curve, under which the state point appears, can be calculated.  The "Lambert W" function is applied to further solve
    the equation in which the unknown appears both within and without the exponential (e) function.

    The end result is the recommended RAS flow rate at which the operator should run the secondary clarifier, given the
    current conditions.  This function will print and return the raw Q_RAS and the rounded Q_RAS for readability, and it
    will also plot the SPA graph in a new figure.
    """

    # Solids concentration at which flux peaks (dy/dx = 0)
    flux_peak = 1 / (alpha + beta*sludge_volume_index)  # g/L

    # Two points to the right of peak for exponential approximation
    x_sp_1, x_sp_2 = map(lambda x: x*flux_peak, (3, 6))  # g/L; solids concentrations
    y_sp_1, y_sp_2 = map(flux, [x_sp_1, x_sp_2], [sludge_volume_index]*2)  # lb/ft2-d; corresponding fluxes

    # Y = A*e^(k*X) --> ln(Y) = k*X + lnA (linearization of exponential function)
    x_arr = np.array([[x_sp_1, 1],
                      [x_sp_2, 1]])  # k coefficients plus lnA constants
    y_arr = np.array([np.log(y_sp_1),
                      np.log(y_sp_2)])  # Natural logarithms of Y values

    k_sp, nat_log_a = np.linalg.solve(x_arr, y_arr)  # L/g, ln(lb/ft2-d); estimates for line
    a = np.exp(nat_log_a)  # lb/ft2-d; exponential coefficient

    # Apply Lambert W function to calculate q_ras
    w_input = -convert(q_in, "mgd") * mixed_liquor_ss / (NUMBER_OF_TANKS*TANK_AREA*a*np.exp(mixed_liquor_ss*k_sp + 1))
    lamb_w = min(map(lambertw, [w_input]*2, [0, -1]))  # W func gives two solutions, take the smaller

    # Calculate operating variable, q_ras
    q_ras = float((q_in * mixed_liquor_ss * k_sp) / lamb_w)  # mgd; convert to Python float
    op_qras = oper_round(q_in, q_ras)  # Round for ease of input, up for conservatism

    # Plot State Point Analysis Graph
    underflow_x_intercept = solids_underflow_calc(q_in, q_ras, mixed_liquor_ss, TANK_AREA, NUMBER_OF_TANKS)[0][1]  # g/L
    solids_max = round(underflow_x_intercept / 5) * 5  # g/L; Max solids conc. on x-axis rounded to multiple of 5

    plot_spa(q_in, op_qras, TANK_AREA, NUMBER_OF_TANKS, sludge_volume_index, mixed_liquor_ss, solids_max=solids_max)

    print(q_ras, op_qras)
    return q_ras, op_qras


# Test Run
alpha, beta, delta, gamma = def_constants(test)
Qras = find_state_point(SVI, MLSS, Qin)[1]
