import alm_module
import gps_time
from datetime import datetime
import numpy as np
import math
from math import atan2, sqrt, degrees, sin, cos, pi, asin

import matplotlib.pyplot as plt
from matplotlib.pyplot import rc, rcParams
import matplotlib.patches as mpatches
from pylab import *

def datetime2list(datetime : datetime):
    return [datetime.year, datetime.month, datetime.day, datetime.hour, datetime.minute, datetime.second]

def latlonh2xyz(latitudeRad : 'float', longitudeRad : 'float', heightMeters : 'float') -> 'np.array':
    a = 6378137
    eSquared = 0.00669438002290
    N = a / math.sqrt(1-(eSquared*(math.sin(latitudeRad)**2)))

    Coords = [
        (N + heightMeters) * math.cos(latitudeRad) * math.cos(longitudeRad),
        (N + heightMeters) * math.cos(latitudeRad) * math.sin(longitudeRad),
        (N * (1 - eSquared) + heightMeters) * math.sin(latitudeRad)
    ]
    return np.array(Coords)

def calculateSatellitePositionAtGivenTime(currentDate : 'datetime', eccentricity : 'float', timeOfAlmanac : 'int', orbitalInclination : 'float', 
                                 rateOfRightAscension : 'float', sqrta : 'float', rightAscensionOfAscendingNode : 'float', 
                                 perigee : 'float', meanAnomaly : 'float', gpsWeek : 'int'):
    '''
    rate of right ascension in deg/1000sec
    '''
    mi =  3.986005 * (10**14)
    perigeeE = 7.2921151467 / (10**5)    

    currentWeek, currentSecond = gps_time.date2tow(datetime2list(currentDate))
    timeElapsed = currentSecond - timeOfAlmanac + (currentWeek - gpsWeek) * 604800

    a = sqrta**2
    n = sqrt(mi/(a**3))
    Mk = np.deg2rad(meanAnomaly) + (n * timeElapsed)

    Ek0 = np.deg2rad(Mk)
    Ek = np.deg2rad(Mk) + eccentricity * sin(Ek0)

    while (abs(Ek - Ek0) > 10 ** (-12)):
        Ek0 = Ek
        Ek = Mk + eccentricity * sin(Ek0)

    vk = atan2(sqrt(1 - (eccentricity ** 2)) * sin(Ek),
                    cos(Ek) - eccentricity)

    phikRad = vk + np.deg2rad(perigee)

    rk = a * (1 - eccentricity * cos(Ek))

    xk = rk * cos(phikRad)
    yk = rk * sin(phikRad)

    adjustedRightAscensionOfAscendingNodeRad = (
        np.deg2rad(rightAscensionOfAscendingNode) + (np.deg2rad(rateOfRightAscension/1000) - perigeeE) * timeElapsed - perigeeE * timeOfAlmanac)
        
    Xk = xk * cos(adjustedRightAscensionOfAscendingNodeRad) - yk * cos(np.deg2rad(orbitalInclination)) * sin(adjustedRightAscensionOfAscendingNodeRad)
    Yk = xk * sin(adjustedRightAscensionOfAscendingNodeRad) + yk * cos(np.deg2rad(orbitalInclination)) * cos(adjustedRightAscensionOfAscendingNodeRad)
    Zk = yk * sin(np.deg2rad(orbitalInclination))

    return [Xk, Yk, Zk]

def calculateSatellitePositionAtGivenTimeFromAlmanac(currentDate : 'datetime', loadedAlmanac):
    eccentricity = loadedAlmanac[2]
    timeOfAlmanac = loadedAlmanac[7]
    nominalInclinationAngle = 54
    orbitalInclination = nominalInclinationAngle + loadedAlmanac[8]
    rateOfRightAscension = loadedAlmanac[9]
    sqrta = loadedAlmanac[3]
    rightAscensionOfAscendingNode = loadedAlmanac[4]
    perigee = loadedAlmanac[5]
    meanAnomaly = loadedAlmanac[6]
    gpsWeek = loadedAlmanac[12]
    position = calculateSatellitePositionAtGivenTime(currentDate=currentDate, eccentricity=eccentricity, timeOfAlmanac=timeOfAlmanac, orbitalInclination=orbitalInclination,
                                                    rateOfRightAscension=rateOfRightAscension, sqrta=sqrta,
                                                    rightAscensionOfAscendingNode=rightAscensionOfAscendingNode,
                                                    perigee=perigee, meanAnomaly=meanAnomaly, gpsWeek=gpsWeek)
    return position

def countVectorsObjectObserver (objectsCoords : 'np.array', observersCoords : 'np.array'):
    vectorsAirpalneAirport = objectsCoords - observersCoords
    return vectorsAirpalneAirport

def countNEU(ovserverLatitudeRad : 'float', observerLongitudeRad : 'float'):
    n = np.array([
        -math.sin(ovserverLatitudeRad)*math.cos(observerLongitudeRad),
        -math.sin(ovserverLatitudeRad)*math.sin(observerLongitudeRad),
        math.cos(ovserverLatitudeRad)
    ])

    e = np.array([
        -math.sin(observerLongitudeRad),
        math.cos(observerLongitudeRad),
        0
    ])

    u = np.array([
        math.cos(ovserverLatitudeRad)*math.cos(observerLongitudeRad),
        math.cos(ovserverLatitudeRad)*math.sin(observerLongitudeRad),
        math.sin(ovserverLatitudeRad)
    ])

    return [n,e,u]

def countRneut(n,e,u):
    Rneu = np.column_stack((n,e,u))
    Rneut = Rneu.transpose()
    
    return Rneut

def countNeuVector(Rneut, vectorObjectObserver):
    neuVector = Rneut @ vectorObjectObserver

    return neuVector

def countDistance(neuVector):
    s = np.sqrt(np.square(neuVector[0]) + np.square(neuVector[1]) + np.square(neuVector[2]))
    return s

def countAz(neuVectorE, neuVectorN):
    Azs = np.arctan2(neuVectorE,neuVectorN)
    if(Azs < 0): Azs += 2 * math.pi

    return Azs

def countHs(neuVector):
    hs = np.arcsin(neuVector[2]/(np.sqrt(np.square(neuVector[0]) + np.square(neuVector[1]) + np.square(neuVector[2]))))

    return hs

def objectElevationAndAzimtuhDistance(objectsXYZ : 'np.array', observersPhiLamH : 'np.array'):
    observersXYZ = latlonh2xyz(*observersPhiLamH)

    vectorObjectObserver = countVectorsObjectObserver(objectsXYZ, observersXYZ)
    n, e, u = countNEU(observersPhiLamH[0], observersPhiLamH[1])
    Rneut = countRneut(n, e, u)
    vectorNeu = countNeuVector(Rneut, vectorObjectObserver)
    elevation, azimuth, distance = countHs(vectorNeu), countAz(vectorNeu[1], vectorNeu[0]), countDistance(vectorNeu)
    return [elevation, azimuth, distance]

def countElementInAMatrix(objectCoordinate : 'float', observerCoordinate : 'float', distanceObjectObserver : 'float'):
    return (-(objectCoordinate - observerCoordinate) / distanceObjectObserver)

def countRowInAMatrix(objectXYZ : 'list[float]', observerXYZ : 'list[float]'):
    distanceObjectObserver = np.linalg.norm(np.array(objectXYZ) - np.array(observerXYZ))
    return [countElementInAMatrix(objectXYZ[0], observerXYZ[0], distanceObjectObserver),
            countElementInAMatrix(objectXYZ[1], observerXYZ[1], distanceObjectObserver),
            countElementInAMatrix(objectXYZ[2], observerXYZ[2], distanceObjectObserver),
            1]

def countQMatrix(AMatrix : 'np.array()'):
    AMatrixTransposed = np.transpose(AMatrix)
    return np.linalg.inv(AMatrixTransposed @ AMatrix)

def countQneuMatrix(QMatrix : 'np.array()', observerLatitudeRad : 'float', observerLongitudeRad : 'float'):
    n, e, u = countNEU(observerLatitudeRad, observerLongitudeRad)
    Rneu = np.column_stack([n,e,u])
    RneuTransposed = np.transpose(Rneu)

    resizedQMatrix = QMatrix[:3, :3]
    Qneu = RneuTransposed @  resizedQMatrix @ Rneu
    return Qneu

def countGDOP(QMatrix : 'np.array'):
    if QMatrix.shape != (4,4):
        raise ValueError
    return sqrt(QMatrix.trace())

def countTDOP(QneuMatrix : 'np.array'):
    if QneuMatrix.shape != (4, 4):
        raise ValueError
    return sqrt(QneuMatrix[3,3])

def countHDOP(QneuMatrix : 'np.array'):
    if QneuMatrix.shape != (3,3):
        raise ValueError
    return sqrt(QneuMatrix[0,0] + QneuMatrix[1,1])

def countVDOP(QneuMatrix : 'np.array'):
    if QneuMatrix.shape != (3,3):
        raise ValueError
    return sqrt(QneuMatrix[2,2])

def countPDOP(QMatrix : 'np.array'):
    if QMatrix.shape != (3,3) and QMatrix.shape != (4,4):
        raise ValueError
    return sqrt(QMatrix[0,0] + QMatrix[1,1] + QMatrix[2,2])

def countDOPs(AMatrix : 'np.array', observerLatitudeRad : 'float', observerLongitudeRad : 'float'):
    '''
    ### Arguments

    - AMatrix : 'np.array' - Matrix A. Number of rows should be equal to the number of rows in satellitesVisibleOverOneEpoch.
    - observerLatitudeRad : 'float' - Float representig observers latitude in radians.
    - observerLongitudeRad : 'float' - Float representig observers longitude in radians.

    ### Returns

    Touple containing:
    
    0. GDOP
    1. PDOP
    2. TDOP
    3. HDOP
    4. VDOP

    '''
    Q = countQMatrix(AMatrix)
    Qneu = countQneuMatrix(Q, observerLatitudeRad, observerLongitudeRad)
    gdop = countGDOP(Q)
    pdop = countPDOP(Q)
    tdop = countTDOP(Q)
    hdop = countHDOP(Qneu)
    vdop = countVDOP(Qneu)
    return (gdop, pdop, tdop, hdop, vdop)

### THE function

def createSatellitesVisibleSatellitesPositions(startDate, endDate, interval, phiDeg, lamDeg, hMeters, maskDeg, almanacFilePath):
    observerLatLonRad = np.array([np.deg2rad(phiDeg), np.deg2rad(lamDeg), hMeters])

    data, prns = alm_module.get_alm_data(almanacFilePath)
    data = np.array(data)

    noIntervals = (endDate - startDate) // interval
    epochs = [startDate + interval * i for i in range(noIntervals)]

    satellitesPositionsForWholePeriod = []
    satellitesVisibleOverWholePeriod = []
    matricesAAndDOPsOverWholePerid = []

    for epoch in epochs:

        A = []
        satellitesPositionsForOneEpoch = []
        satellitesVisibleOverOneEpoch = []

        for i, satellite in enumerate(data):
            position = calculateSatellitePositionAtGivenTimeFromAlmanac(epoch, satellite)

            satellitesPositionsForOneEpoch.append(position)
            elevation, azimuth, distance = objectElevationAndAzimtuhDistance(np.array(position), observerLatLonRad)

            satellitesVisibleOverOneEpoch.append([i, prns[i], elevation, azimuth, distance])
            if (np.rad2deg(elevation) > maskDeg):

                observersXYZ = latlonh2xyz(*observerLatLonRad)
                newrow = countRowInAMatrix(position, observersXYZ)
                A.append(newrow)
        
        A = np.vstack(A)
        DOPs = countDOPs(A, observerLatLonRad[0], observerLatLonRad[1])

        satellitesPositionsForWholePeriod.append(satellitesPositionsForOneEpoch)
        satellitesVisibleOverWholePeriod.append(satellitesVisibleOverOneEpoch)
        matricesAAndDOPsOverWholePerid.append((A, DOPs))
    satellitesPositionsForWholePeriod = np.array(satellitesPositionsForWholePeriod)

    return prns, satellitesPositionsForWholePeriod, satellitesVisibleOverWholePeriod, matricesAAndDOPsOverWholePerid 

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot3dSatellitesOnAxis(axis, satellitesPositionsForWholePeriod : 'list[list[list]]', prns : 'list[str]', showLegend : 'bool' = False, colorScheme : 'str' = "b"):
    noColors = len(satellitesPositionsForWholePeriod[0])
    colorMap = get_cmap(noColors + 1)
    axis.cla()
    epochsSatelitesPositions = np.array(satellitesPositionsForWholePeriod)
    for i, satellite in enumerate([epochsSatelitesPositions[:,i] for i in range(len(epochsSatelitesPositions[0]))]):
        X, Y, Z = zip(*satellite)
        axis.plot(X, Y, Z, c=colorMap(i), label=prns[i])
        
    r = 6370_000
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    axis.plot_surface(x, y, z, color=colorScheme)
    
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")

    if showLegend: axis.legend(ncol=4)

def plotSkyplotOnAxis(axis, satellitesElevationsAzimuthsOverTime, fontsize = 12):
    axis.cla()

    rc('grid', color='gray', linewidth=1, linestyle='--')
    rc('xtick', labelsize = fontsize)
    rc('ytick', labelsize = fontsize)
    rc('font', size = fontsize)
    
    green   ='#467821'
    blue    ='#348ABD'
    red     ='#A60628'
    orange  ='#E24A33'
    purple  ='#7A68A6'
    yellow  ='#ffff00'
    black   ='#000000'

    axis.set_theta_zero_location('N')
    axis.set_theta_direction(-1)
    
    satellitesForLater = dict()

    for i, sat_positions in enumerate(satellitesElevationsAzimuthsOverTime):
        PG = 0 
        for (PRN, el, az) in sat_positions: 
            if PRN[0] == 'G': 
                satColor = blue
            elif PRN[0] == 'R': 
                satColor = red
            elif PRN[0] == 'E': 
                satColor = orange
            elif PRN[0] == 'Q': 
                satColor = purple
            elif PRN[0] == 'I': 
                satColor = black
            elif PRN[0] == 'C': 
                satColor = yellow
            elif PRN[0] == 'S': 
                satColor = green
            PG += 1
            
            if(i == len(satellitesElevationsAzimuthsOverTime) - 1):
                axis.annotate(PRN, 
                            xy=(np.radians(az), 90-el),
                            bbox=dict(boxstyle="round", fc = green, alpha = 0.1),
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            color = 'k')
                axis.scatter(np.radians(az), 90-el, color = satColor, label=PRN[0])
            
            if PRN in satellitesForLater:
                satellitesForLater[PRN].append([np.radians(az), 90-el, satColor])
            else:
                satellitesForLater[PRN] = [[np.radians(az), 90-el, satColor]]
    
    for satellite in satellitesForLater.values():
        azs, zenitals, colors = zip(*satellite)
        axis.plot(azs, zenitals, color=colors[0]) 
               

    gnss     = mpatches.Patch(color=green,  label='{:02.0f}  GNSS'.format(PG))
    chiny    = mpatches.Patch(color=yellow,  label='Chinese beidou')
    rosja    = mpatches.Patch(color=red,  label='Russian glonos')
    europa    = mpatches.Patch(color=orange,  label='European galileo')
    ameryka    = mpatches.Patch(color=blue,  label='American gps')
    indie    = mpatches.Patch(color=black,  label='Indian')
    japonia    = mpatches.Patch(color=purple,  label='Japanese qzss')
    axis.legend(handles=[gnss, chiny, rosja, europa, ameryka, indie, japonia], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    

    axis.set_yticks(range(0, 90+10, 10))
    yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
    axis.set_yticklabels(yLabel)

      
def plotSkyplotFromSatVisibleOnAxis(axis, satellitesVisibleOverWholePeriod, startEpochNo : 'int', numberOfEpochs : 'int' = 0, fontsize : 'int' = 12,
                gps : 'bool' = True, glonos : 'bool' = True, galileo : 'bool' = True, 
                qzss : 'bool' = True, irnss : 'bool' = True, beidou : 'bool' = True, other : 'bool' = True):
    if startEpochNo + numberOfEpochs > len(satellitesVisibleOverWholePeriod):
        return False
    plotTable = []
    for i in range(startEpochNo, startEpochNo + numberOfEpochs):
        epochTable = []
        originalTable = satellitesVisibleOverWholePeriod[i]
        for i, satellite in enumerate(originalTable):
            if not galileo and satellite[1][0] == 'E':
                continue
            if not gps and satellite[1][0] == 'G':
                continue
            if not beidou and satellite[1][0] == 'C':
                continue
            if not glonos and satellite[1][0] == 'R':
                continue
            if not qzss and satellite[1][0] == 'Q':
                continue
            if not irnss and satellite[1][0] == 'I':
                continue
            if not other and satellite[1][0] == 'S':
                continue

            epochTable.append([satellite[1], np.rad2deg(satellite[2]), np.rad2deg(satellite[3])])
        plotTable.append(epochTable)

    plotSkyplotOnAxis(axis, plotTable, fontsize=fontsize)

def xyz2latlon(xs, ys, zs):
    """
    function which converts ECEF satellite position position to latitude and longitude
    Based on rvtolatlong.m in Richard Rieber's orbital library on mathwork.com

    Note: that this is only suitable for groundtrack visualization, not rigorous
    calculations.
    """
    R = [xs, ys, zs]
    r_delta = np.linalg.norm(R[0:1]);
    sinA = R[1]/r_delta;
    cosA = R[0]/r_delta;

    lon = atan2(sinA,cosA)

    if lon < - pi:
        lon = lon + 2 * pi;
    lat = asin(R[2]/np.linalg.norm(R))
    return [degrees(lat), degrees(lon)]

def plotGroundtrackOnAxis(prn, latLonGround, axis, coastlineDataPath : 'str', fontsize : 'int' = 20):
    '''
    Ploting satellite ground track:
        Satellite groundtrack longitude and latitude are calculated 
        from XYZ(ECEF) satellites bythe use of function(above): groundtrack_latlon(xs, ys, zs)
    
    INPUT:
        lon_lat_ground ; list of list [lon, lat]
        Coastline.txt - file with coastline coordinate
    '''
    rc('grid', color='gray', linewidth=0.1, linestyle='--')
    rc('xtick', labelsize = fontsize)
    rc('ytick', labelsize = fontsize)
    rc('font', size = fontsize)
    params = {'legend.fontsize': 8,  'legend.handlelength': 2}
    rcParams.update(params)
    
    coastline_data= np.loadtxt(coastlineDataPath, skiprows=1)
    
    green   ='#467821'
    blue    ='#348ABD'
    red     ='#A60628'
    orange  ='#E24A33'
    purple  ='#7A68A6'
    yellow  ='#ffff00'
    black   ='#000000'

    if prn[0] == 'G':
        satColor = blue
    elif prn[0] == 'R':
        satColor = red
    elif prn[0] == 'E':
        satColor = orange
    elif prn[0] == 'Q':
        satColor = purple
    elif prn[0] == 'I':
        satColor = black
    elif prn[0] == 'C':
        satColor = yellow
    elif prn[0] == 'S':
        satColor = green

    axis.plot(coastline_data[:,0],coastline_data[:,1],'g', linewidth=0.5);
    axis.set_xlabel('Longitude $[\mathrm{^\circ}]$',fontsize=14)
    axis.set_ylabel('Latitude $[\mathrm{^\circ}]$',fontsize=14)
    axis.set_yticks(np.arange(-90, 91, 10))
    axis.set_xticks(np.arange(-180, 181, 15))
    axis.set_xbound(-180, 181)
    axis.set_ybound(-90, 91)

    latsGround, lonsGround = zip(*latLonGround)
    axis.annotate(prn, 
                xy=(lonsGround[-1], latsGround[-1]),
                bbox=dict(boxstyle="round", fc = green, alpha = 0.1),
                horizontalalignment='center',
                verticalalignment='bottom',
                color = 'k')
    
    axis.plot(lonsGround, latsGround, linewidth = 1, color = satColor)
    axis.scatter(lonsGround[-1], latsGround[-1], s = 2, color = satColor)
     
    chiny    = mpatches.Patch(color=yellow,  label='Chinese beidou')
    rosja    = mpatches.Patch(color=red,  label='Russian glonos')
    europa    = mpatches.Patch(color=orange,  label='European galileo')
    ameryka    = mpatches.Patch(color=blue,  label='American gps')
    indie    = mpatches.Patch(color=black,  label='Indian')
    japonia    = mpatches.Patch(color=purple,  label='Japanese qzss')

    axis.legend(handles=[chiny, rosja, europa, ameryka, indie, japonia], loc='upper left', borderaxespad=0.)
    axis.grid(True)

def satellitesPositionsForWholePeriod2satellitesPhiLam(satellitesPositionsForWholePeriod):
    satellitesPhiLam = []
    noSatellites = len(satellitesPositionsForWholePeriod[0])
    for i in range(noSatellites):
        satellite = []
        for epoch in satellitesPositionsForWholePeriod:
            satellite.append(epoch[i])
        satellitesPhiLam.append(satellite)

    for satellite in satellitesPhiLam:
        for i in range(len(satellite)):
            x,y,z = satellite[i]
            satellite[i] = xyz2latlon(x,y,z)
    
    return satellitesPhiLam

def plotVisibleSatellitesNumberOnAxis(axis, satellitesVisibleOverWholePeriod):
    axis.cla()
    x = range(1, len(satellitesVisibleOverWholePeriod) + 1)
    y = [h for h in [len(epoch) for epoch in satellitesVisibleOverWholePeriod]]
    axis.bar(x, y, 0.8, 0)
    axis.grid()
    axis.set_yticks(np.arange(0, max(y) + 1, 1))
    axis.set_xticks(np.arange(0,x[-1],x[-1]//15 + 1))
    axis.set_xlabel("Epoch's number")
    axis.set_ylabel("Number of Visible Satellites")

def plotDOPsOnAxis(axis, DOPsForWholePeriod):
    x = range(len(DOPsForWholePeriod))
    dops = list(zip(*DOPsForWholePeriod))
    if len(dops) != 5:
        return False
    axis.cla()
    colors = ['#0CAAEB', '#1641F2', '#5E1FDB', '#D516F2', '#ED406D']
    dopsNames = ["GDOP", "PDOP", "TDOP", "HDOP", "VDOP"]
    for i in range(len(dops)):
        axis.plot(x, dops[i], color=colors[i], linewidth=2)
        axis.scatter(x, dops[i], color=colors[i], s=15, label=dopsNames[i])
    axis.legend()
    axis.grid()
    axis.set_yticks(np.arange(0,1,0.05))
    axis.set_xticks(np.arange(0,x[-1] + 1,len(x) // 10))
    axis.set_xlabel("Epoch's number")
    axis.set_ylabel("Value")
    
def plotElevationsOnAxis(axis, satellitesVisibleOverWholePeriod, legend=False):
    axis.cla()
    satellites = dict()

    for epoch in satellitesVisibleOverWholePeriod:
        for satellite in epoch:
            prn = satellite[1]
            elevation = satellite[2]

            if prn in satellites.keys():
                satellites[prn].append(round(np.rad2deg(elevation), 3))
            else:
                satellites[prn] = [round(np.rad2deg(elevation),3)]

    noColors = len(satellites.keys())
    colorMap = get_cmap(noColors+1)
    
    for i, key in enumerate(satellites.keys()):
        x = range(len(satellites[key]))
        y = satellites[key]
        axis.plot(x, y, label=key, c = colorMap(i))
    
    if(legend):
        axis.legend(ncols=3)

    axis.set_xticks(np.arange(0,len(satellitesVisibleOverWholePeriod),len(satellitesVisibleOverWholePeriod)//15 + 1))
    axis.set_yticks(np.arange(-90,90,10))
    axis.set_xlabel("Epoch's number")
    axis.set_ylabel("Elevation [°]")
    
    axis.grid()

# Uncoment visualizations

# startDate = datetime.datetime(2023,2,23,0,0,0)
# endDate = datetime.datetime(2023,2,23,23,59,59)
# interval = timedelta(0,1200)

# phi = 52
# lam = 21
# h = 100
# mask = 10 # stopni
# folder = pathlib.Path(__file__).parent.resolve()

# prns, satellitesPositionsForWholePeriod, satellitesVisibleOverWholePeriod, matricesAAndDOPsOverWholePerid = createSatellitesVisibleSatellitesPositions(
#     startDate, endDate, interval, phi, lam, h, mask, folder / "Almanac.alm"
# )

# noSatellites = len(prns)

# # plot3dSatellites(satellitesPositionsForWholePeriod, noColors=noSatellites)
# # plotSkyplot(satellitesVisibleOverWholePeriod, gps=False, beidou=False, glonos=False, qzss=False, )

# satellitesPhiLam = satellitesPositionsForWholePeriod2satellitesPhiLam(satellitesPositionsForWholePeriod)
# # define figure
# w, h = plt.figaspect(0.5)
# fig = plt.figure(figsize=(w,h))
# ax = fig.add_subplot(1,1,1)
# path = pathlib.Path(__file__).parent.resolve()/'Coastline.txt'

# plotDOPsOnAxis(ax, [x[1] for x in matricesAAndDOPsOverWholePerid])
# # plotElevationsOnAxis(ax, satellitesVisibleOverWholePeriod)
# plt.show()


# import wget
# file = 'ftp://ftp.trimble.com/pub/eph/Almanac.alm'
# wget.download(file,nazwa_wynikowego_pliku)