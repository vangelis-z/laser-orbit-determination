#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Orbit determination example

CAUTION: this is only an example, there is no guarantee that all computations are 100% correct, nor that this example won't break with future Orekit versions.
"""

import os
from datetime import datetime
from datetime import timedelta


import jdk4py
import orekit_jpype as orekit
# Initializing Orekit and JVM
if "JAVA_HOME" not in os.environ:
    os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)

orekit.initVM()
orekit.pyhelpers.setup_orekit_data(from_pip_library=True)


import getpass
from spacetrack import SpaceTrackClient

import numpy as np
import pandas as pd

import plotly.io as pio
import plotly.graph_objs as go

from orekit_jpype.pyhelpers import datetime_to_absolutedate
from orekit_jpype.pyhelpers import absolutedate_to_datetime
from orekit_jpype.pyhelpers import JArray_double2D

from org.orekit.utils import Constants as orekit_constants
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.files.sinex import SinexLoader, Station
from org.orekit.data import DataSource
from org.orekit.estimation.measurements import GroundStation
from org.orekit.frames import TopocentricFrame

from org.orekit.propagation.analytical.tle import TLE
from org.orekit.attitudes import FrameAlignedProvider
from org.orekit.propagation.analytical.tle import SGP4
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder
from org.orekit.propagation.conversion import NumericalPropagatorBuilder
from org.orekit.orbits import PositionAngleType

from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import ThirdBodyAttraction

from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
from org.orekit.forces.radiation import SolarRadiationPressure

from org.orekit.forces.gravity import Relativity

from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
from org.orekit.models.earth.atmosphere import NRLMSISE00
# from org.orekit.forces.drag.atmosphere import DTM2000
from org.orekit.forces.drag import IsotropicDrag
from org.orekit.forces.drag import DragForce

from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer
from org.orekit.estimation.leastsquares import BatchLSEstimator

from org.orekit.estimation.measurements import Range, AngularAzEl, ObservableSatellite
from org.orekit.models.earth.troposphere import MendesPavlisModel
from org.orekit.estimation.measurements.modifiers import RangeTroposphericDelayModifier

from org.orekit.frames import LocalOrbitalFrame
from org.orekit.frames import LOFType

from org.orekit.utils import CartesianDerivativesFilter
from org.hipparchus.linear import Array2DRowRealMatrix

from org.hipparchus.geometry.euclidean.threed import Vector3D



from slrDataUtils import SlrDlManager
from ccsdsUtils import Ccsds



# OD parameters
# First, some parameters need to be defined for the orbit determination:
#   * Satellite ID in NORAD, COSPAR and SIC code format.
#     These IDs can be found here: https://edc.dgfi.tum.de/en/satellites/
#   * Spacecraft mass: important for the drag term
#   * Measurement weights: used to weight certain measurements more than others during the orbit
#     estimation.  Here, we only have range measurements and we do not know the confidence
#     associated to these measurements, so all weights are identical.
#   * OD date: date at which the orbit will be estimated.
#   * Data collection duration: for example, if equals 2 days, the laser data from the 2 days before
#     the OD date will be used to estimate the orbit.  This value is an important trade-off for the
#     quality of the orbit determination:
#     * The longer the duration, the more ranging data is available, which can increase the quality
#       of the estimation
#     * The longer the duration, the longer the orbit must be propagated, and the higher the
#       covariance because of the orbit perturbations such as the gravity field, drag, Sun, Moon,
#       etc.

# Satellite parameters
sat_list = {
    'envisat': {
        'norad_id': 27386,  # For Space-Track TLE queries
        'cospar_id': '0200901',  # For laser ranging data queries
        'sic_id': '6179',  # For writing in CPF files
        'mass': 8000.0, # kg; TODO: compute proper value
        'cross_section': 100.0, # m2; TODO: compute proper value
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'lageos2': {
        'norad_id': 22195,
        'cospar_id': '9207002',
        'sic_id': '5986',
        'mass': 405.0, # kg
        'cross_section': 0.2827, # m2
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'technosat': {
        'norad_id': 42829,
        'cospar_id': '1704205',
        'sic_id': '6203',
        'mass': 20.0, # kg
        'cross_section': 0.10, # m2,
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'snet1': {
        'norad_id': 43189,
        'cospar_id': '1801410',
        'sic_id': '6204',
        'mass': 8.0, # kg
        'cross_section': 0.07,
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    }
}
sc_name = 'lageos2'  # Change the name to select a different satellite in the dict

# Orbit determination parameters
"""
NPT: Normal point data. Recommended option. The data is pre-filtered by the laser data providers
FRD: Full-rate data. Warning, these are a lot of data points (potentially tens of thousands per day),
    the execution time could be greatly increased
"""
laser_data_type = 'NPT'

range_weight = 1.0 # Will be normalized later (i.e divided by the number of observations)
range_sigma = 1.0 # Estimated covariance of the range measurements, in meters

az_weight = 0.1  # Do not weigh the Az/El measurements too much because they are much less accurate than ranges
el_weight = 0.1
az_sigma = float(np.deg2rad(0.01))
el_sigma = float(np.deg2rad(0.01))

odDate = datetime(2019, 12, 5) # Beginning of the orbit determination
collectionDuration = 2 # days
startCollectionDate = odDate + timedelta(days=-collectionDuration)

# Orbit propagator parameters
prop_min_step = 0.001 # s
prop_max_step = 300.0 # s
prop_position_error = 10.0 # m

# Estimator parameters
estimator_position_scale = 1.0 # m
estimator_convergence_thres = 1e-2
estimator_max_iterations = 25
estimator_max_evaluations = 35


# ## API credentials
# The following sets up accounts for SpaceTrack (for orbit data) and the EDC API (for laser ranging data).
# * A SpaceTrack account is required, it can be created for free at: https://www.space-track.org/auth/createAccount
# * An EDC account is required, it can be created for free at: https://edc.dgfi.tum.de/en/register/


# Space-Track
identity_st = input('Enter SpaceTrack username')
password_st = getpass.getpass(prompt='Enter SpaceTrack password for account {}'.format(identity_st))
st = SpaceTrackClient(identity=identity_st, password=password_st)


# EDC API
username_edc = input('Enter EDC API username')
password_edc = getpass.getpass(prompt='Enter EDC API password for account {}'.format(username_edc)) # You will get prompted for your password
url = 'https://edc.dgfi.tum.de/api/v1/'


# ## Setting up models
# Setting up Orekit frames and models
tod = FramesFactory.getTOD(IERSConventions.IERS_2010, False) # Taking tidal effects into account when interpolating EOP parameters
gcrf = FramesFactory.getGCRF()
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
# Selecting frames to use for OD
eci = gcrf
ecef = itrf

wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(ecef)
moon = CelestialBodyFactory.getMoon()
sun = CelestialBodyFactory.getSun()

utc = TimeScalesFactory.getUTC()
mjd_utc_epoch = AbsoluteDate(1858, 11, 17, 0, 0, 0.0, utc)


# Import station data from file
stationFile = 'SLRF2020_POS+VEL_2023.10.02.snx'
stationEccFile = 'ecc_xyz.snx'
stations_map = SinexLoader(DataSource(stationFile)).getStations()
ecc_map = SinexLoader(DataSource(stationEccFile)).getStations()


# Converting the station data to Orekit GroundStation objects
station_keys = stations_map.keySet()

n_errors = 0
station_df = pd.DataFrame(columns=['lat_deg', 'lon_deg', 'alt_m', 'GroundStation'])
for key in station_keys:
    station_data = stations_map.get(key)
    ecc_data = ecc_map.get(key)
    if ecc_data.getEccRefSystem() != Station.ReferenceSystem.XYZ:
        print('Error, eccentricity coordinate system not XYZ')

    epoch_velocity = station_data.getEpoch()
    durationSinceEpoch = datetime_to_absolutedate(odDate).durationFrom(epoch_velocity)  # seconds

    # Computing current station position using velocity data
    station_pos_at_epoch = station_data.getPosition()
    vel = station_data.getVelocity()  # m/s
    station_pos_current = station_pos_at_epoch.add(vel.scalarMultiply(durationSinceEpoch))

    # Adding eccentricity
    try:
        station_pos_current = station_pos_current.add(ecc_data.getEccentricities(datetime_to_absolutedate(odDate)))
        # Converting to ground station object
        geodeticPoint = wgs84Ellipsoid.transform(station_pos_current, itrf, datetime_to_absolutedate(odDate))
        lon_deg = np.rad2deg(geodeticPoint.getLongitude())
        lat_deg = np.rad2deg(geodeticPoint.getLatitude())
        alt_m = geodeticPoint.getAltitude()
        topocentricFrame = TopocentricFrame(wgs84Ellipsoid, geodeticPoint, key)
        groundStation = GroundStation(topocentricFrame)
        station_df.loc[key] = [lat_deg, lon_deg, alt_m, groundStation]
    except:
        # And exception is thrown when the odDate is not in the date range of the eccentricity entry for this station
        # This is simply for stations which do not exist anymore at odDate
        n_errors += 1

station_df = station_df.sort_index()
print(station_df)


# The orbit determination needs a first guess. For this, we use Two-Line Elements. Retrieving the latest TLE prior to the beginning of the orbit determination. It is important to have a "fresh" TLE, because the newer the TLE, the better the orbit estimation.
rawTle = st.tle(norad_cat_id=sat_list[sc_name]['norad_id'], epoch='<{}'.format(odDate), orderby='epoch desc', limit=1, format='tle')
tleLine1 = rawTle.split('\n')[0]
tleLine2 = rawTle.split('\n')[1]
print(tleLine1)
print(tleLine2)


# Setting up the propagator from the initial TLEs
orekitTle = TLE(tleLine1, tleLine2)
pointing = FrameAlignedProvider(eci)
sgp4Propagator = SGP4(orekitTle, pointing, sat_list[sc_name]['mass'])

tleInitialState = sgp4Propagator.getInitialState()
tleEpoch = tleInitialState.getDate()
tleOrbit_TEME = tleInitialState.getOrbit()
tlePV_ECI = tleOrbit_TEME.getPVCoordinates(eci)

tleOrbit_ECI = CartesianOrbit(tlePV_ECI, eci, wgs84Ellipsoid.getGM())

integratorBuilder = DormandPrince853IntegratorBuilder(prop_min_step, prop_max_step, prop_position_error)

propagatorBuilder = NumericalPropagatorBuilder(tleOrbit_ECI,
                                               integratorBuilder, PositionAngleType.MEAN, estimator_position_scale)
propagatorBuilder.setMass(sat_list[sc_name]['mass'])
propagatorBuilder.setAttitudeProvider(pointing)


# Adding perturbation forces to the propagator
# Earth gravity field with degree 64 and order 64
gravityProvider = GravityFieldFactory.getNormalizedProvider(64, 64)
gravityAttractionModel = HolmesFeatherstoneAttractionModel(ecef, gravityProvider)
propagatorBuilder.addForceModel(gravityAttractionModel)

# Moon and Sun perturbations
moon_3dbodyattraction = ThirdBodyAttraction(moon)
propagatorBuilder.addForceModel(moon_3dbodyattraction)
sun_3dbodyattraction = ThirdBodyAttraction(sun)
propagatorBuilder.addForceModel(sun_3dbodyattraction)

# Solar radiation pressure
isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(sat_list[sc_name]['cross_section'], sat_list[sc_name]['cr'])
solarRadiationPressure = SolarRadiationPressure(sun, wgs84Ellipsoid,
                                                isotropicRadiationSingleCoeff)
propagatorBuilder.addForceModel(solarRadiationPressure)

# Relativity
relativity = Relativity(orekit_constants.EIGEN5C_EARTH_MU)
propagatorBuilder.addForceModel(relativity)


# Adding atmospheric drag to the propagator
# Atmospheric drag
cswl = CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt")

atmosphere = NRLMSISE00(cswl, sun, wgs84Ellipsoid)
#atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)
isotropicDrag = IsotropicDrag(sat_list[sc_name]['cross_section'], sat_list[sc_name]['cd'])
dragForce = DragForce(atmosphere, isotropicDrag)
propagatorBuilder.addForceModel(dragForce)


# Setting up the estimator
matrixDecomposer = QRDecomposer(1e-11)
optimizer = GaussNewtonOptimizer(matrixDecomposer, False)

estimator = BatchLSEstimator(optimizer, propagatorBuilder)
estimator.setParametersConvergenceThreshold(estimator_convergence_thres)
estimator.setMaxIterations(estimator_max_iterations)
estimator.setMaxEvaluations(estimator_max_evaluations)


# ## Fetching range data
# Looking for laser ranging data prior to the OD date.
slr_dl_manager = SlrDlManager(username_edc=username_edc,
                     password_edc=password_edc)

laserDatasetList = slr_dl_manager.querySlrData(laser_data_type,
                                                sat_list[sc_name]['cospar_id'],
                                                startCollectionDate, odDate)
print(laserDatasetList)


# Downloading the list of observations to NPT files (in a folder in the gitignore, we don't need to manage these files in git).
slrDataFrame = slr_dl_manager.dlAndParseSlrData(laser_data_type, laserDatasetList, 'slr-data')
print(slrDataFrame)


# Adding the measurements to the estimator.
#
# Update 2024-05-13: The Az/el measurements are commented out but are kept for documentation purposes.
observableSatellite = ObservableSatellite(0) # Propagator index = 0

for receiveTime, slrData in slrDataFrame.iterrows():
    if slrData['station-id'] in station_df.index: # Checking if station exists in the stations list, because it might not be up-to-date
        if not np.isnan(slrData['range']):  # If this data point contains a valid range measurement
            orekitRange = Range(station_df.loc[slrData['station-id'], 'GroundStation'],
                                True, # Two-way measurement
                                receiveTime,
                                slrData['range'],
                                range_sigma,
                                range_weight,
                                observableSatellite
                               ) # Uses date of signal reception; https://www.orekit.org/static/apidocs/org/orekit/estimation/measurements/Range.html

            range_tropo_delay_modifier = RangeTroposphericDelayModifier(
                MendesPavlisModel(slrData['temperature_K'],
                                  slrData['pressure_mbar'],
                                  slrData['humidity'],
                                  slrData['wavelength_microm'])
            )
            orekitRange.addModifier(range_tropo_delay_modifier)

            estimator.addMeasurement(orekitRange)
        #if not np.isnan(slrData['az']):  # If this data point contains a valid angles measurement
        #    orekitAzEl = AngularAzEl(station_df.loc[slrData['station-id'], 'GroundStation'],
        #                            receiveTime,
        #                            JArray('double')([slrData['az'], slrData['el']]),
        #                            JArray('double')([az_sigma, el_sigma]),
        #                            JArray('double')([az_weight, el_weight]),
        #                            observableSatellite)
        #    estimator.addMeasurement(orekitAzEl)


# Performing the OD
# Estimate the orbit. This step can take a long time.
estimatedPropagatorArray = estimator.estimate()


# Propagating the estimated orbit
dt = 300.0
date_start = datetime_to_absolutedate(startCollectionDate)
date_start = date_start.shiftedBy(-86400.0)
date_end = datetime_to_absolutedate(odDate)
date_end = date_end.shiftedBy(86400.0) # Stopping 1 day after OD date

# First propagating in ephemeris mode
estimatedPropagator = estimatedPropagatorArray[0]
estimatedInitialState = estimatedPropagator.getInitialState()
actualOdDate = estimatedInitialState.getDate()
estimatedPropagator.resetInitialState(estimatedInitialState)
eph_generator = estimatedPropagator.getEphemerisGenerator()

# Propagating from 1 day before data collection
# To 1 week after orbit determination (for CPF generation)
estimatedPropagator.propagate(date_start, datetime_to_absolutedate(odDate).shiftedBy(7 * 86400.0))
bounded_propagator = eph_generator.getGeneratedEphemeris()


# Covariance analysis
# Creating the LVLH frame, computing the covariance matrix in both TOD and LVLH frames
# Creating the LVLH frame
# It must be associated to the bounded propagator, not the original numerical propagator
lvlh = LocalOrbitalFrame(eci, LOFType.LVLH, bounded_propagator, 'LVLH')

# Getting covariance matrix in ECI frame
covMat_eci_java = estimator.getPhysicalCovariances(1.0e-10)

# Converting matrix to LVLH frame
# Getting an inertial frame aligned with the LVLH frame at this instant
# The LVLH is normally not inertial, but this should not affect results too much
# Reference: David Vallado, Covariance Transformations for Satellite Flight Dynamics Operations, 2003
eci2lvlh_frozen = eci.getTransformTo(lvlh, actualOdDate).freeze()

# Computing Jacobian
jacobianDoubleArray = JArray_double2D(np.zeros((6, 6)))
eci2lvlh_frozen.getJacobian(CartesianDerivativesFilter.USE_PV, jacobianDoubleArray)
jacobian = Array2DRowRealMatrix(jacobianDoubleArray)
# Applying Jacobian to convert matrix to lvlh
covMat_lvlh_java = jacobian.multiply(
    covMat_eci_java.multiply(jacobian.transpose()))

# Converting the Java matrices to numpy
covarianceMat_eci = np.matrix([covMat_eci_java.getRow(iRow)
                              for iRow in range(0, covMat_eci_java.getRowDimension())])
covarianceMat_lvlh = np.matrix([covMat_lvlh_java.getRow(iRow)
                              for iRow in range(0, covMat_lvlh_java.getRowDimension())])


# Computing the position and velocity standard deviation
pos_std_crossTrack = np.sqrt(covarianceMat_lvlh[0,0])
pos_std_alongTrack = np.sqrt(covarianceMat_lvlh[1,1])
pos_std_outOfPlane = np.sqrt(covarianceMat_lvlh[2,2])
print(f'Position std: cross-track {pos_std_crossTrack:.3e} m, along-track {pos_std_alongTrack:.3e} m, out-of-plane {pos_std_outOfPlane:.3e} m')

vel_std_crossTrack = np.sqrt(covarianceMat_lvlh[3,3])
vel_std_alongTrack = np.sqrt(covarianceMat_lvlh[4,4])
vel_std_outOfPlane = np.sqrt(covarianceMat_lvlh[5,5])
print(f'Velocity std: cross-track {vel_std_crossTrack:.3e} m/s, along-track {vel_std_alongTrack:.3e} m/s, out-of-plane {vel_std_outOfPlane:.3e} m/s')


# ## CCSDS OPM
# Writing a CCSDS OPM message
sat_properties = {
     'mass': sat_list[sc_name]['mass'],
     'solar_rad_area': sat_list[sc_name]['cross_section'],
     'solar_rad_coeff': sat_list[sc_name]['cd'],
     'drag_area': sat_list[sc_name]['cross_section'],
     'drag_coeff': sat_list[sc_name]['cr']
}

ccsds_writer = Ccsds(originator='GOR', object_name=sc_name, object_id=sat_list[sc_name]['norad_id'], sat_properties=sat_properties)

pv_eci_init = estimatedInitialState.getPVCoordinates()
pos_eci_init = np.array(pv_eci_init.getPosition().toArray())
vel_eci_init = np.array(pv_eci_init.getVelocity().toArray())

ccsds_writer.write_opm('OPM.txt', absolutedate_to_datetime(actualOdDate), pos_eci_init, vel_eci_init, covarianceMat_eci, 'EARTH', 'GCRF')


# ## Analyzing residuals
# Getting the estimated and measured ranges.
propagatorParameters   = estimator.getPropagatorParametersDrivers(True)
measurementsParameters = estimator.getMeasurementsParametersDrivers(True)

lastEstimations = estimator.getLastEstimations()
valueSet = lastEstimations.values()
estimatedMeasurements = valueSet.toArray()
keySet = lastEstimations.keySet()
realMeasurements = keySet.toArray()


range_residuals = pd.DataFrame(columns=['range'])
azel_residuals = pd.DataFrame(columns=['az', 'el'])

for estMeas, realMeas in zip(estimatedMeasurements, realMeasurements):
    #estMeas = EstimatedMeasurement.cast_(estMeas)
    estimatedValue = estMeas.getEstimatedValue()
    pyDateTime = absolutedate_to_datetime(estMeas.getDate())

    if isinstance(realMeas, Range):
        observedValue = realMeas.getObservedValue()
        range_residuals.loc[pyDateTime] = np.array(observedValue) - np.array(estimatedValue)
    elif isinstance(realMeas, AngularAzEl):
        observedValue = realMeas.getObservedValue()
        azel_residuals.loc[pyDateTime] = np.array(observedValue) - np.array(estimatedValue)

print(range_residuals)
print(azel_residuals)


# Setting up Plotly for offline mode
# pio.renderers.default = 'jupyterlab+png'  # Uncomment for interactive plots
pio.renderers.default = 'png'


# Plotting range residuals
trace = go.Scattergl(
    x=range_residuals.index, y=range_residuals['range'],
    mode='markers',
    name='Range'
)

data = [trace]

layout = go.Layout(
    title = 'Range residuals',
    xaxis = dict(
        title = 'Datetime UTC'
    ),
    yaxis = dict(
        title = 'Range residual (m)'
    )
)

fig = dict(data=data, layout=layout)

pio.show(fig)


# Plotting angles residuals (if available)
trace_az = go.Scattergl(
    x=azel_residuals.index, y=np.rad2deg(azel_residuals['az']),
    mode='markers',
    name='Azimuth'
)

trace_el = go.Scattergl(
    x=azel_residuals.index, y=np.rad2deg(azel_residuals['el']),
    mode='markers',
    name='Elevation'
)

data = [trace_az, trace_el]

layout = go.Layout(
    title = 'Angle residuals',
    xaxis = dict(
        title = 'Datetime UTC'
    ),
    yaxis = dict(
        title = 'Angle residual (deg)'
    )
)

fig = dict(data=data, layout=layout)

pio.show(fig)


# ## Comparison with CPF
# The EDC API also provides Consolidated Prediction Files, which contain spacecraft position/velocity in ITRF frame as generated by their orbit determination system. We can compare our orbit determination with the one from the latest CPF prior to the first ranging data used in our orbit determination.

# Requesting CPF data
cpfList = slr_dl_manager.queryCpfData(
                       sat_list[sc_name]['cospar_id'], startCollectionDate - timedelta(days=1))
print(cpfList)


# Downloading and parsing CPF data
cpfDataFrame = slr_dl_manager.dlAndParseCpfData(
                                 [cpfList.index[0]], # If several ephemerides are available for this day, only take the first
                                 startCollectionDate - timedelta(days=1),
                                 odDate + timedelta(days=1))
print(cpfDataFrame)


# ## Propagating the solution
# Propagating the solution and:
# * Saving the PV coordinates from both the solution and the initial TLE guess.
# * Computing the difference in LVLH frame between the solution and the initial TLE guess.
# * Computing the difference in LVLH frame between the solution and the CPF file.
# Propagating the bounded propagator to retrieve the intermediate states

deltaPV_tle_lvlh_dict = {}
deltaPV_cpf_lvlh_dict = {}

date_current = date_start
while date_current.compareTo(date_end) <= 0:
    datetime_current = absolutedate_to_datetime(date_current)
    spacecraftState = bounded_propagator.propagate(date_current)

    '''
    When getting PV coordinates using the SGP4 propagator in LVLH frame,
    it is actually a "delta" from the PV coordinates resulting from the orbit determination
    because this LVLH frame is centered on the satellite's current position based on the orbit determination
    '''
    deltaPV_lvlh = sgp4Propagator.getPVCoordinates(date_current, lvlh)
    deltaPV_tle_lvlh_dict[datetime_current] = [deltaPV_lvlh.getPosition().getX(),
                                                 deltaPV_lvlh.getPosition().getY(),
                                                 deltaPV_lvlh.getPosition().getZ(),
                                                 deltaPV_lvlh.getPosition().getNorm(),
                                                 deltaPV_lvlh.getVelocity().getX(),
                                                 deltaPV_lvlh.getVelocity().getY(),
                                                 deltaPV_lvlh.getVelocity().getZ(),
                                                 deltaPV_lvlh.getVelocity().getNorm()]

    pos_cpf_ecef = cpfDataFrame.loc[datetime_current]
    ecef2lvlh = ecef.getStaticTransformTo(lvlh, date_current)
    delta_pos_cpf_lvlh_vector = ecef2lvlh.transformPosition(Vector3D(float(pos_cpf_ecef['x']),
                                                                     float(pos_cpf_ecef['y']),
                                                                     float(pos_cpf_ecef['z'])))
    deltaPV_cpf_lvlh_dict[datetime_current] = [delta_pos_cpf_lvlh_vector.getX(),
                                                 delta_pos_cpf_lvlh_vector.getY(),
                                                 delta_pos_cpf_lvlh_vector.getZ(),
                                                 delta_pos_cpf_lvlh_vector.getNorm()]

    date_current = date_current.shiftedBy(dt)

deltaPV_tle_lvlh_df = pd.DataFrame.from_dict(deltaPV_tle_lvlh_dict,
                                             columns=['x', 'y', 'z', 'pos_norm', 'vx', 'vy', 'vz', 'vel_norm'],
                                             orient='index')

deltaPV_cpf_lvlh_df = pd.DataFrame.from_dict(deltaPV_cpf_lvlh_dict,
                                             columns=['x', 'y', 'z', 'norm'],
                                             orient='index')


# ## Plotting difference between estimated orbit and CPF

# Plotting position difference. The grey area represents the time window where range measurements were used to perform the orbit determination.
# Rectangles to visualise time window for orbit determination.

od_window_rectangle =  {
    'type': 'rect',
    # x-reference is assigned to the x-values
    'xref': 'x',
    # y-reference is assigned to the plot paper [0,1]
    'yref': 'paper',
    'x0': startCollectionDate,
    'y0': 0,
    'x1': odDate,
    'y1': 1,
    'fillcolor': '#d3d3d3',
    'opacity': 0.3,
    'line': {
        'width': 0,
    }
}

traceX = go.Scattergl(
    x = deltaPV_cpf_lvlh_df.index,
    y = deltaPV_cpf_lvlh_df['x'],
    mode='lines',
    name='Cross-track'
)

traceY = go.Scattergl(
    x = deltaPV_cpf_lvlh_df.index,
    y = deltaPV_cpf_lvlh_df['y'],
    mode='lines',
    name='Along track'
)

traceZ = go.Scattergl(
    x = deltaPV_cpf_lvlh_df.index,
    y = deltaPV_cpf_lvlh_df['z'],
    mode='lines',
    name='Out-of-plane'
)

data = [traceX, traceY, traceZ]

layout = go.Layout(
    title = 'Delta position between CPF and estimation in LVLH frame',
    xaxis = dict(
        title = 'Datetime UTC'
    ),
    yaxis = dict(
        title = 'Position difference (m)'
    ),
    shapes=[od_window_rectangle]
)

fig = dict(data=data, layout=layout)

pio.show(fig)


# ## Writing own CPF file

# A CPF file usually contains 7 days of orbit prediction in ECEF frame with a sample time of 5 minutes, to allow the laser stations to track the satellite.
#
# Therefore we have to propagate for 7 days.
# Function to compute MJD days and seconds of day
def datetime_to_mjd_days_seconds(le_datetime):
    apparent_clock_offset_s = datetime_to_absolutedate(le_datetime).offsetFrom(
        mjd_utc_epoch, utc)
    days_since_mjd_epoch = int(np.floor(apparent_clock_offset_s / 86400.0))
    seconds_of_day = apparent_clock_offset_s - days_since_mjd_epoch * 86400.0
    return days_since_mjd_epoch, seconds_of_day

date_end_cpf = datetime_to_absolutedate(odDate).shiftedBy(7 * 86400.0)

PV_ecef_cpf_dict = {}

dt = 300.0
date_current = datetime_to_absolutedate(odDate)
while date_current.compareTo(date_end_cpf) <= 0:
    datetime_current = absolutedate_to_datetime(date_current)
    spacecraftState = bounded_propagator.propagate(date_current)

    PV_ecef_cpf = spacecraftState.getPVCoordinates(ecef)
    pos_ecef_cpf = PV_ecef_cpf.getPosition()
    vel_ecef_cpf = PV_ecef_cpf.getVelocity()
    PV_ecef_cpf_dict[datetime_current] = [
        pos_ecef_cpf.getX(),
        pos_ecef_cpf.getY(),
        pos_ecef_cpf.getZ(),
        vel_ecef_cpf.getX(),
        vel_ecef_cpf.getY(),
        vel_ecef_cpf.getZ()
    ]

    date_current = date_current.shiftedBy(dt)

PV_ecef_cpf_df = pd.DataFrame.from_dict(
    PV_ecef_cpf_dict,
    columns=['x', 'y', 'z', 'vx', 'vy', 'vz'],
    orient='index'
)
PV_ecef_cpf_df['DateTimeUTC'] = PV_ecef_cpf_df.index
PV_ecef_cpf_df['mjd_days'], PV_ecef_cpf_df['seconds_of_day'] = zip(*PV_ecef_cpf_df['DateTimeUTC'].apply(lambda x:
                                                                         datetime_to_mjd_days_seconds(x)))

slr_dl_manager.write_cpf(cpf_df=PV_ecef_cpf_df,
          cpf_filename='cpf_out.ass',
          ephemeris_source='ASS',
          production_date=odDate,
          ephemeris_sequence=5999,
          target_name=sc_name,
          cospar_id=sat_list[sc_name]['cospar_id'],
          sic=sat_list[sc_name]['sic_id'],
          norad_id=str(sat_list[sc_name]['norad_id']),
          ephemeris_start_date=odDate,
          ephemeris_end_date=absolutedate_to_datetime(date_end_cpf),
          step_time=int(dt))


# ## Comparison with TLE

# Plotting the components of the position different between the TLE and the estimation, in LVLH frame. The grey area represents the time window where range measurements were used to perform the orbit determination.
traceX = go.Scattergl(
    x = deltaPV_tle_lvlh_df['x'].index,
    y = deltaPV_tle_lvlh_df['x'],
    mode='lines',
    name='Cross-Track'
)

traceY = go.Scattergl(
    x = deltaPV_tle_lvlh_df['y'].index,
    y = deltaPV_tle_lvlh_df['y'],
    mode='lines',
    name='Along-Track'
)

traceZ = go.Scattergl(
    x = deltaPV_tle_lvlh_df['z'].index,
    y = deltaPV_tle_lvlh_df['z'],
    mode='lines',
    name='Out-Of-Plane'
)

data = [traceX, traceY, traceZ]

layout = go.Layout(
    title = 'Delta position between TLE and estimation in LVLH frame',
    xaxis = dict(
        title = 'Datetime UTC'
    ),
    yaxis = dict(
        title = 'Position difference (m)'
    ),
    shapes=[od_window_rectangle]
)

fig = dict(data=data, layout=layout)

pio.show(fig)
