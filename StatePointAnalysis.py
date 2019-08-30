# Required Libraries #
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from mpmath import lambertw


# Functions #
def convert(val, orig_units):
    """orig_units: m-g/h-L, lb/MG, MG-g/d-ft2-L, mgd"""
    
    if orig_units == "m-g/h-L":
        conv = 1/1000*100**3/3.28084**2*24/1000*2.20462
    elif orig_units == "lb/MG":
        conv = 1/1000000*7.48*3.28084**3/100**3*1000*1000/2.20462
    elif orig_units == "MG-g/d-ft2-L":
        conv = 1000000/1000*1000000/3.28084**3/7.48*2.20462/1000
    elif orig_units == "mgd":
        conv = 1 / convert(1, "lb/MG")
    else:
        conv = 1/(-val)

    return val*conv

def defConstants(test):
    """SVISN, SVISS, SVIGN, SVIGS"""

    #Test: (alpha, beta, delta, gamma)
    testDict = {
        "SVISN": (0.261, 0.00170, 0.00370, 14.9),
        "SVISS": (0.211, 0.00236, 0.00593, 14.6),
        "SVIGN": (0.351, 0.00058, 0.00602, 18.2),
        "SVIGS": (0.245, 0.00296, 0.01073, 24.3)
        }

    constants = testDict[test]

    return constants

def surfOR(Qin, areaTank, numTank, SVI):
    x1, y1 = 0, 0

    y2 = 1.25*gamma*np.exp(-(delta*SVI+1)) / (alpha + beta*SVI)
    y2 = convert(y2, "m-g/h-L")
    
    x2 = y2*numTank*areaTank / Qin
    x2 = convert(x2, "lb/MG")

    x = (x1, x2)
    y = (y1, y2)

    return x, y

def solidsUR(Qin, Qras, MLSS, areaTank, numTank):
    x1, y2 = 0, 0

    y1 = (Qin+Qras)*MLSS / (numTank*areaTank)
    y1 = convert(y1, "MG-g/d-ft2-L")

    x2 = y1*numTank*areaTank / Qras
    x2 = convert(x2, "lb/MG")

    x = (x1, x2)
    y = (y1, y2)

    return x, y

def flux(solidsConc, SVI):
    y = gamma*solidsConc*np.exp(-(delta*SVI+(alpha+beta*SVI)*solidsConc))
    y = convert(y, "m-g/h-L")

    return y

def settleFlux(SVI, solidsMax = 15):
    x = np.linspace(0, solidsMax, num=100)
    y = flux(x, SVI)

    return x, y

def plotSPA(Qin, Qras, areaTank, numTank, SVI, MLSS, solidsMax = 15):
    xOR, yOR = surfOR(Qin, areaTank, numTank, SVI)
    xUR, yUR = solidsUR(Qin, Qras, MLSS, areaTank, numTank)
    xFlux, yFlux = settleFlux(SVI, solidsMax)

    SOR = yOR[1]/xOR[1]
    statePt = SOR*MLSS

    plt.plot(xOR, yOR, "g", xUR, yUR, "orange", xFlux, yFlux, "b")
    plt.plot(MLSS, statePt, "ro")
    plt.axis([0, solidsMax, 0, 60])
    plt.grid(True)

    plt.title("State Point Analysis")
    plt.xlabel("Solids Concentration (g/L)")
    plt.ylabel("Solids Flux (lb/ft2-d)")

    rasSub = r'$Q_{RAS}$'
    txt = "%s = %s MGD" % (rasSub, Qras)
    plt.text(0.5*solidsMax, 2*gamma, txt, size=16, backgroundcolor="white", \
             bbox=dict(facecolor="white"))

    plt.show()

def operRound(ref, val):
    if ref < 10:
        precision = 0.10
    elif 10 <= ref < 100:
        precision = 0.25
    else:
        precision = 0.50

    opQras = np.ceil(val/precision)*precision

    return opQras

def findStatePoint(SVI, MLSS, Qin):
    #Solids concentration at which flux peaks
    fluxPeak = 1 / (alpha + beta*SVI) #g/L

    #Two points to the right of peak for exponential approximation
    x1, x2 = map(lambda x: x*fluxPeak, (3,6)) #g/L; solids concentrations
    y1, y2 = map(flux, [x1, x2], [SVI]*2) #lb/ft2-d; corresponding fluxes

    #Y = A*e^(k*X) --> ln(Y) = k*X + lnA (linearization of exponential function)
    xs = np.array([[x1, 1],
                   [x2, 1]]) #k coefficients plus lnA constants
    ys = np.array([np.log(y1),
                   np.log(y2)]) #Natural logarithms of Y values

    k, lnA = np.linalg.solve(xs, ys) #L/g, ln(lb/ft2-d); estimates for line
    A = np.exp(lnA) #lb/ft2-d; exponential coefficient

    #Apply Lambert W function to calculate Qras
    wInput = -convert(Qin, "mgd")*MLSS / (numTank*areaTank*A*np.exp(MLSS*k + 1))
    lambW = min(map(lambertw, [wInput]*2, [0, -1])) #W func gives two solutions

    #Calculate operating variable, Qras
    Qras = float((Qin * MLSS * k) / lambW) #mgd; convert to Python float
    opQras = operRound(Qin, Qras) #Round for ease of input, up for conservatism

    #Plot State Point Analysis Graph
    xURInt = solidsUR(Qin, Qras, MLSS, areaTank, numTank)[0][1] #g/L; UR x-int
    solidsMax = round(xURInt / 5) * 5 #g/L; Max solids conc. on x-axis

    plotSPA(Qin, opQras, areaTank, numTank, SVI, MLSS, solidsMax=solidsMax)

    print(Qras, opQras)
    return Qras, opQras


# Constants #
numTank = 20 #Number of clarifiers
areaTank = 19760 #ft2; Area of each clarifier (length x width)


### INPUTS ###
test = "SVISN" #Test used to determine sludge volume index
SVI  = 100     #mL/g; Sludge Volume Index as determined by "test"
MLSS = 1.5     #g/L; Mixed Liquor Suspended Solids concentration
Qin  = 165     #mgd; Influent flow
##############

alpha, beta, delta, gamma = defConstants(test)
Qras = findStatePoint(SVI, MLSS, Qin)[1]


    
xOR, yOR = surfOR(Qin, areaTank, numTank, SVI)
xUR, yUR = solidsUR(Qin, Qras, MLSS, areaTank, numTank)
xFlux, yFlux = settleFlux(SVI)
SOR = yOR[1]/xOR[1]
statePt = SOR*MLSS

plt.plot(xOR, yOR, "g", xUR, yUR, "orange", xFlux, yFlux, "b")
plt.plot(MLSS, statePt, "ro")

plt.axis([0, 15, 0, 60])
plt.grid(True)
plt.title("State Point Analysis")
plt.xlabel("Solids Concentration (g/L)")
plt.ylabel("Solids Flux (lb/ft2-d)")

fluxPeak = 1 / (alpha + beta*SVI)
yPeak = flux(fluxPeak, SVI)

plt.plot((fluxPeak, fluxPeak), (0, yPeak), "b--")
plt.annotate("peak flux", xy=(fluxPeak, yPeak), xytext=(fluxPeak, yPeak+5),
             arrowprops=dict(facecolor="blue", shrink=0.1, width=4))
plt.annotate(r'$solids_{peak}$', xy=(fluxPeak, 0), xytext=(fluxPeak+0.25, 7),
             arrowprops=dict(facecolor="blue", shrink=0.1, width=4))

plt.plot((MLSS, MLSS), (0, statePt), "r--")
plt.annotate("state point", xy=(MLSS, statePt), xytext=(MLSS-1, statePt+4),
             arrowprops=dict(facecolor="red", shrink=0.1, width=4))
plt.annotate("MLSS", xy=(MLSS, 0), xytext=(MLSS+1, 4),
             arrowprops=dict(facecolor="red", shrink=0.1, width=4))

plt.annotate("RAS conc.", xy=(xUR[1], 0), xytext=(xUR[1]+0.1, 5),
             arrowprops=dict(facecolor="orange", shrink=0.1, width=4))
plt.annotate("SLR", xy=(0, yUR[0]), xytext=(0.5, yUR[0]+6),
             arrowprops=dict(facecolor="orange", shrink=0.1, width=4))

x1, x2 = map(lambda x: x*fluxPeak, (3,6))
y1, y2 = map(flux, [x1, x2], [SVI]*2)

xs = np.array([[x1, 1], [x2, 1]])
ys = np.array([np.log(y1),np.log(y2)])

k, lnA = np.linalg.solve(xs, ys)
A = np.exp(lnA)

exX = np.linspace(fluxPeak+1, 15, num=100)
exY = A*np.exp(k*exX)

plt.plot(exX, exY, "b--")
plt.annotate("exponential approximation", xy=(4, A*np.exp(k*4)),
             xytext=(5, A*np.exp(k*4)+5),
             arrowprops=dict(facecolor="blue", shrink=0.1, width=4,
                             headwidth=4), style="italic", weight="semibold")
plt.annotate("surface overflow rate", xy=(xOR[1], yOR[1]),
             xytext=(xOR[1]+2, yOR[1]+4),
             arrowprops=dict(facecolor="green", shrink=0.1, width=4,
                             headwidth=4), style="italic", weight="semibold")
plt.annotate("solids underflow rate", xy=(7,
            convert(-Qras/(areaTank*numTank)*7 \
            + MLSS*(Qin+Qras)/(numTank*areaTank), "MG-g/d-ft2-L")),
             xytext=(7.5, 22),
             arrowprops=dict(facecolor="orange", shrink=0.1, width=4,
                             headwidth=4), style="italic", weight="semibold")
plt.annotate("settling flux curve", xy=(fluxPeak+1, flux(fluxPeak+1, SVI)),
             xytext=(fluxPeak+3, yPeak),
             arrowprops=dict(facecolor="blue", shrink=0.1, width=4,
                             headwidth=4), style="italic", weight="semibold")

plt.show()


