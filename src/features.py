import numpy as np


def extract_features(sample):

    df = sample["data"]

    features = {}


    for column in df.columns:

        values = df[column].values


        features[column+"_mean"] = np.mean(values)

        features[column+"_std"] = np.std(values)

        features[column+"_max"] = np.max(values)

        features[column+"_min"] = np.min(values)

        features[column+"_range"] = (
            np.max(values) -
            np.min(values)
        )

        features[column+"_energy"] = (
            np.sum(values ** 2)
        )


    # combined motion features

    accel = np.sqrt(
        df.accel_x**2 +
        df.accel_y**2 +
        df.accel_z**2
    )


    gyro = np.sqrt(
        df.gyro_x**2 +
        df.gyro_y**2 +
        df.gyro_z**2
    )


    features["acceleration_peak"] = accel.max()

    features["acceleration_mean"] = accel.mean()

    features["gyro_peak"] = gyro.max()

    features["gyro_mean"] = gyro.mean()


    features["samples"] = len(df)


    return features