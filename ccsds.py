from datetime import datetime
from typing import Dict, List


class CCSDS:
    originator = ''
    object_name = ''
    object_id = ''
    sat_properties = {
        'mass': 0.0,
        'srp_area': 0.0,
        'srp_coeff': 0.0,
        'drag_area': 0.0,
        'drag_coeff': 0.0
    }

    def __init__(
        self, originator: str, object_name: str, object_id: str, sat_properties: Dict
    ) -> None:
        """
        Default constructor

            :param originator: creating agency
            :param object_name: common name of object
            :param object_id: preferably COSPAR ID or similar
            :param sat_properties: dict containing at least:
                - mass: float, kg
                - srp_area: float, area for solar radiation pressure in sq. m
                - srp_coeff: float, solar radiation pressure coefficient
                - drag_area: float, area for drag in sq. m
                - drag_coeff: float, drag coefficient
        """
        self.originator = originator
        self.object_name = object_name
        self.object_id = object_id
        self.sat_properties = sat_properties

    def write_opm(
        self,
        filename: str,
        epoch: datetime,
        pos_array: List[float],
        vel_array: List[float],
        cov_matrix: List[List[float]],
        center_name: str,
        frame_name: str
    ) -> None:
        """
        Write Orbit Parameter Message (OPM) to a file.
        Timescale is forced to UTC.

            :param filename: output filename
            :param epoch: OPM epoch
            :param pos_array: 3-array of floats, m
            :param vel_array: 3-array of floats, m/s
            :param cov_matrix: NOT USED
            :param center_name: origin of reference frame
            :param frame_name: raference frame
        """

        epoch = f'{epoch:%Y-%m-%dT%H:%M:%S.%f}'
        pos_array = 1e-3 * pos_array
        vel_array = 1e-3 * vel_array

        with open(filename, 'w') as fh:
            # header
            fh.write('CCSDS_OPM_VERS = 2.0\n\n')
            fh.write(f'CREATION_DATE = {datetime.utcnow():%Y-%m-%dT%H:%M:%S}\n')
            fh.write(f'ORIGINATOR = {self.originator}\n\n')
            # metadata
            fh.write(f'OBJECT_NAME = {self.object_name}\n')
            fh.write(f'OBJECT_ID = {self.object_id}\n')
            fh.write(f'CENTER_NAME = {center_name}\n')
            fh.write(f'REF_FRAME = {frame_name}\n')
            fh.write('TIME_SYSTEM = UTC\n')
            fh.write('\n')
            fh.write('COMMENT  Orbit determination based on SLR data\n\n')
            # data
            fh.write('COMMENT  State vector\n')
            fh.write(f'EPOCH = {epoch_str[:-3]}\n')
            fh.write(f'X = {pos_km[0]:.9f}  [km]\n')
            fh.write(f'Y = {pos_km[1]:.9f}  [km]\n')
            fh.write(f'Z = {pos_km[2]:.9f}  [km]\n')
            fh.write(f'X_DOT = {vel_km_s[0]:.12f}  [km/s]\n')
            fh.write(f'Y_DOT = {vel_km_s[1]:.12f}  [km/s]\n')
            fh.write(f'Z_DOT = {vel_km_s[2]:.12f}  [km/s]\n')
            fh.write('\n')
            fh.write('COMMENT  Spacecraft parameters\n')
            fh.write(f'MASS = {self.sat_properties["mass"]:.6f}  [kg]\n')
            fh.write(f'SOLAR_RAD_AREA = {self.sat_properties["srp_area"]:.6f}  [m**2]\n')
            fh.write(f'SOLAR_RAD_COEFF = {self.sat_properties["srp_coeff"]:.6f}\n')
            fh.write(f'DRAG_AREA = {self.sat_properties["drag_area"]:.6f}  [m**2]\n')
            fh.write(f'DRAG_COEFF = {self.sat_properties["drag_coeff"]:.6f}\n')
            fh.write('\n')
