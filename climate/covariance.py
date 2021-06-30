from math import radians, sin, cos, asin, sqrt, pi, atan, atan2, fabs
import numpy as np


def _great_circle_distance(point1, point2):
    lat1, lon1 = point1[0], point1[1]
    lat2, lon2 = point2[0], point2[1]

    dlon = fabs(lon1 - lon2)
    dlat = fabs(lat1 - lat2)

    numerator = sqrt(
        (cos(lat2)*sin(dlon))**2 +
        ((cos(lat1)*sin(lat2)) - (sin(lat1)*cos(lat2)*cos(dlon)))**2)

    denominator = (
        (sin(lat1)*sin(lat2)) +
        (cos(lat1)*cos(lat2)*cos(dlon)))

    c = atan2(numerator, denominator)
    return c

great_circle_distance = np.vectorize(_great_circle_distance, signature='(n),(n)->()')
