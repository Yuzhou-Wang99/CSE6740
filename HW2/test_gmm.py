import numpy as np
import unittest
from matplotlib import image
import matplotlib.cm as cm

from submission import (
    initialize_parameters,
    compute_sigma,
    prob,
    E_step,
    M_step,
    likelihood,
    train_model,
    default_convergence,
)


def image_to_matrix(image_file, grays=False):
    """
    Convert .png image to matrix
    of values.
    params:
    image_file = str
    grays = Boolean
    returns:
    img = (color) np.ndarray[np.ndarray[np.ndarray[float]]]
    or (grayscale) np.ndarray[np.ndarray[float]]
    """
    img = image.imread(image_file)
    # in case of transparency values
    if len(img.shape) == 3 and img.shape[2] > 3:
        height, width, depth = img.shape
        new_img = np.zeros([height, width, 3])
        for r in range(height):
            for c in range(width):
                new_img[r, c, :] = img[r, c, 0:3]
        img = np.copy(new_img)
    if grays and len(img.shape) == 3:
        height, width = img.shape[0:2]
        new_img = np.zeros([height, width])
        for r in range(height):
            for c in range(width):
                new_img[r, c] = img[r, c, 0]
        img = new_img
    return img


def matrix_to_image(image_matrix, image_file):
    """
    Convert matrix of color/grayscale
    values  to .png image
    and save to file.

    params:
    image_matrix = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        or (grayscale) numpy.ndarray[numpy.ndarray[float]]
    image_file = str
    """
    # provide cmap to grayscale images
    c_map = None
    if len(image_matrix.shape) < 3:
        c_map = cm.Greys_r
    image.imsave(image_file, image_matrix, cmap=c_map)


def flatten_image_matrix(image_matrix):
    """
    Flatten image matrix from
    Height by Width by Depth
    to (Height*Width) by Depth
    matrix.

    params:
    image_matrix = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        or (grayscale) numpy.ndarray[numpy.ndarray[float]]

    returns:
    flattened_values = (color) numpy.ndarray[numpy.ndarray[float]]
    or (grayscale) numpy.ndarray[float]
    """
    if len(image_matrix.shape) == 3:
        height, width, depth = image_matrix.shape
    else:
        height, width = image_matrix.shape
        depth = 1
    flattened_values = np.zeros([height * width, depth])
    for i, r in enumerate(image_matrix):
        for j, c in enumerate(r):
            flattened_values[i * width + j, :] = c
    return flattened_values


def unflatten_image_matrix(image_matrix, width):
    """
    Unflatten image matrix from
    (Height*Width) by Depth to
    Height by Width by Depth matrix.

    params:
    image_matrix = (color) numpy.ndarray[numpy.ndarray[float]]
    or (grayscale) numpy.ndarray[float]
    width = int

    returns:
    unflattened_values =
        (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        or (grayscale) numpy.ndarray[numpy.ndarray[float]]
    """
    heightWidth = image_matrix.shape[0]
    height = int(heightWidth / width)
    if len(image_matrix.shape) > 1:
        depth = image_matrix.shape[-1]
        unflattened_values = np.zeros([height, width, depth])
        for i in range(height):
            for j in range(width):
                unflattened_values[i, j, :] = image_matrix[i * width + j, :]
    else:
        depth = 1
        unflattened_values = np.zeros([height, width])
        for i in range(height):
            for j in range(width):
                unflattened_values[i, j] = image_matrix[i * width + j]
    return unflattened_values


def image_difference(image_values_1, image_values_2):
    """
    Calculate the total difference
    in values between two images.
    Assumes that both images have same
    shape.

    params:
    image_values_1 = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        or (grayscale) numpy.ndarray[numpy.ndarray[float]]
    image_values_2 = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        or (grayscale) numpy.ndarray[numpy.ndarray[float]]

    returns:
    dist = int
    """
    flat_vals_1 = flatten_image_matrix(image_values_1)
    flat_vals_2 = flatten_image_matrix(image_values_2)
    n, depth = flat_vals_1.shape
    dist = 0.0
    point_thresh = 0.005
    for i in range(n):
        if depth > 1:
            new_dist = sum(abs(flat_vals_1[i] - flat_vals_2[i]))
            if new_dist > depth * point_thresh:
                dist += new_dist
        else:
            new_dist = abs(flat_vals_1[i] - flat_vals_2[i])
            if new_dist > point_thresh:
                dist += new_dist
    return dist


def default_convergence(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (
        abs(prev_likelihood) * 0.9 < abs(new_likelihood) < abs(prev_likelihood) * 1.1
    )

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


def print_success_message():
    print("UnitTest passed successfully!")


def generate_test_mixture(data_size, means, variances, mixing_coefficients):
    """
    Generate synthetic test
    data for a GMM based on
    fixed means, variances and
    mixing coefficients.

    params:
    data_size = (int)
    means = [float]
    variances = [float]
    mixing_coefficients = [float]

    returns:
    data = np.array[float]
    """

    data = np.zeros(data_size)

    indices = np.random.choice(len(means), len(data), p=mixing_coefficients)

    for i in range(len(indices)):
        val = np.random.normal(means[indices[i]], variances[indices[i]])
        while val <= 0:
            val = np.random.normal(means[indices[i]], variances[indices[i]])
        data[i] = val

    return data


class GMMTests(unittest.TestCase):
    def runTest(self):
        pass

    def test_gmm_initialization(self, initialize_parameters=initialize_parameters):
        """Testing the GMM method
        for initializing the training"""
        image_file = "images/Starry.png"
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        m, n = image_matrix.shape
        num_components = 5
        np.random.seed(0)
        means, variances, mixing_coefficients = initialize_parameters(
            image_matrix, num_components
        )
        self.assertTrue(
            variances.shape == (num_components, n, n),
            msg="Incorrect variance dimensions",
        )
        self.assertTrue(
            means.shape == (num_components, n), msg="Incorrect mean dimensions"
        )
        for mean in means:
            self.assertTrue(
                any(np.equal(image_matrix, mean).all(1)),
                msg=("Means should be points from given array"),
            )
        self.assertTrue(
            mixing_coefficients.sum() == 1,
            msg="Incorrect mixing coefficients, make all coefficient sum to 1",
        )
        print_success_message()

    def test_gmm_covariance(self, compute_sigma=compute_sigma):
        """Testing implementation of covariance matrix
        computation explicitly"""
        image_file = "images/Starry.png"
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        m, n = image_matrix.shape
        num_components = 5
        MU = np.array(
            [
                [0.64705884, 0.7490196, 0.7058824],
                [0.98039216, 0.3019608, 0.14509805],
                [0.3764706, 0.39215687, 0.28627452],
                [0.2784314, 0.26666668, 0.23921569],
                [0.16078432, 0.15294118, 0.30588236],
            ]
        )
        SIGMA = np.array(
            [
                [
                    [0.14120309, 0.13409922, 0.07928442],
                    [0.13409922, 0.13596143, 0.09358084],
                    [0.07928442, 0.09358084, 0.09766863],
                ],
                [
                    [0.44409867, -0.04886889, -0.20206978],
                    [-0.04886889, 0.08191175, 0.09531033],
                    [-0.20206978, 0.09531033, 0.18705386],
                ],
                [
                    [0.0587372, 0.05115941, 0.01780809],
                    [0.05115941, 0.06062889, 0.05254236],
                    [0.01780809, 0.05254236, 0.10531252],
                ],
                [
                    [0.0649982, 0.06846332, 0.04307953],
                    [0.06846332, 0.09466889, 0.08934892],
                    [0.04307953, 0.08934892, 0.12813057],
                ],
                [
                    [0.09788626, 0.11438698, 0.0611304],
                    [0.11438698, 0.15272257, 0.09879004],
                    [0.0611304, 0.09879004, 0.09711219],
                ],
            ]
        )

        self.assertTrue(
            np.allclose(SIGMA, compute_sigma(image_matrix, MU)),
            msg="Incorrect covariance matrix.",
        )
        print_success_message()

    def test_gmm_prob(self, prob=prob):
        """Testing the GMM method
        for calculating the probability
        of a given point belonging to a
        component.
        returns:
        prob = float
        """

        image_file = "images/Starry.png"
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        m, n = image_matrix.shape
        mean = np.array([0.0627451, 0.10980392, 0.54901963])
        covariance = np.array(
            [
                [0.28756526, 0.13084501, -0.09662368],
                [0.13084501, 0.11177602, -0.02345659],
                [-0.09662368, -0.02345659, 0.11303925],
            ]
        )
        # Single Input
        p = prob(image_matrix[0], mean, covariance)
        self.assertEqual(
            round(p, 5),
            0.98605,
            msg="Incorrect probability value returned for single input.",
        )

        # Multiple Input
        p = prob(image_matrix[0:5], mean, covariance)
        self.assertEqual(
            list(np.round(p, 5)),
            [0.98605, 0.78737, 1.20351, 1.35478, 0.73028],
            msg="Incorrect probability value returned for multiple input.",
        )

        print_success_message()

    def test_gmm_e_step(self, E_step=E_step):
        """Testing the E-step implementation

        returns:
        r = numpy.ndarray[numpy.ndarray[float]]
        """
        image_file = "images/Starry.png"
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        num_components = 5
        m, n = image_matrix.shape
        means = np.array(
            [
                [0.34901962, 0.3647059, 0.30588236],
                [0.9882353, 0.3254902, 0.19607843],
                [1.0, 0.6117647, 0.5019608],
                [0.37254903, 0.3882353, 0.2901961],
                [0.3529412, 0.40784314, 1.0],
            ]
        )
        covariances = np.array(
            [
                [
                    [0.13715639, 0.03524152, -0.01240736],
                    [0.03524152, 0.06077217, 0.01898307],
                    [-0.01240736, 0.01898307, 0.07848206],
                ],
                [
                    [0.3929004, 0.03238055, -0.10174976],
                    [0.03238055, 0.06016063, 0.02226048],
                    [-0.10174976, 0.02226048, 0.10162983],
                ],
                [
                    [0.40526569, 0.18437279, 0.05891556],
                    [0.18437279, 0.13535137, 0.0603222],
                    [0.05891556, 0.0603222, 0.09712359],
                ],
                [
                    [0.13208355, 0.03362673, -0.01208926],
                    [0.03362673, 0.06261538, 0.01699577],
                    [-0.01208926, 0.01699577, 0.08031248],
                ],
                [
                    [0.13623408, 0.03036055, -0.09287403],
                    [0.03036055, 0.06499729, 0.06576895],
                    [-0.09287403, 0.06576895, 0.49017089],
                ],
            ]
        )
        pis = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        r = E_step(image_matrix, means, covariances, pis, num_components)
        expected_r_rows = np.array(
            [
                35184.26013053,
                12110.51997221,
                19475.93046123,
                33416.32214795,
                16778.96728808,
            ]
        )
        self.assertEqual(
            round(r.sum()),
            m,
            msg="Incorrect responsibility values, sum of all elements must be equal to m.",
        )
        self.assertTrue(
            np.allclose(r.sum(axis=0), 1),
            msg="Incorrect responsibility values, columns are not normalized.",
        )
        self.assertTrue(
            np.allclose(r.sum(axis=1), expected_r_rows),
            msg="Incorrect responsibility values, rows are not normalized.",
        )
        print_success_message()

    def test_gmm_m_step(self, M_step=M_step):
        """Testing the M-step implementation

        returns:
        pi = numpy.ndarray[]
        mu = numpy.ndarray[numpy.ndarray[float]]
        sigma = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        """
        image_file = "images/Starry.png"
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        num_components = 3

        r = np.array(
            [
                [
                    0.51660555,
                    0.52444999,
                    0.50810777,
                    0.51151982,
                    0.4997758,
                    0.51134715,
                    0.4997758,
                    0.49475051,
                    0.48168621,
                    0.47946386,
                ],
                [
                    0.10036031,
                    0.09948503,
                    0.1052672,
                    0.10687822,
                    0.11345191,
                    0.10697943,
                    0.11345191,
                    0.11705775,
                    0.11919758,
                    0.12314451,
                ],
                [
                    0.38303414,
                    0.37606498,
                    0.38662503,
                    0.38160197,
                    0.3867723,
                    0.38167342,
                    0.3867723,
                    0.38819173,
                    0.39911622,
                    0.39739164,
                ],
            ]
        )
        mu, sigma, pi = M_step(image_matrix[:10], r, num_components)
        expected_PI = np.array([0.50274825, 0.11052739, 0.38672437])
        expected_MU = np.array(
            [
                [0.15787668, 0.22587548, 0.23974434],
                [0.15651327, 0.22400117, 0.23191456],
                [0.1576726, 0.2254149, 0.23655895],
            ]
        )
        expected_SIGMA = np.array(
            [
                [
                    [0.01099723, 0.0115452, 0.00967741],
                    [0.0115452, 0.01219342, 0.01038057],
                    [0.00967741, 0.01038057, 0.01508434],
                ],
                [
                    [0.01020192, 0.010746, 0.00888965],
                    [0.010746, 0.01139497, 0.00961631],
                    [0.00888965, 0.00961631, 0.01457653],
                ],
                [
                    [0.01070972, 0.01125898, 0.00943508],
                    [0.01125898, 0.01191069, 0.01015814],
                    [0.00943508, 0.01015814, 0.01503744],
                ],
            ]
        )
        self.assertTrue(
            np.allclose(pi, expected_PI), msg="Incorrect new coefficient matrix."
        )
        self.assertTrue(np.allclose(mu, expected_MU), msg="Incorrect new means matrix.")
        self.assertTrue(
            np.allclose(sigma, expected_SIGMA), msg="Incorrect new covariance matrix."
        )
        print_success_message()

    def test_gmm_likelihood(self, likelihood=likelihood):
        """Testing the GMM method
        for calculating the overall
        model probability.
        Should return -46437.

        returns:
        likelihood = float
        """

        image_file = "images/Starry.png"
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        num_components = 5
        m, n = image_matrix.shape
        means = np.array(
            [
                [0.34901962, 0.3647059, 0.30588236],
                [0.9882353, 0.3254902, 0.19607843],
                [1.0, 0.6117647, 0.5019608],
                [0.37254903, 0.3882353, 0.2901961],
                [0.3529412, 0.40784314, 1.0],
            ]
        )
        covariances = np.array(
            [
                [
                    [0.13715639, 0.03524152, -0.01240736],
                    [0.03524152, 0.06077217, 0.01898307],
                    [-0.01240736, 0.01898307, 0.07848206],
                ],
                [
                    [0.3929004, 0.03238055, -0.10174976],
                    [0.03238055, 0.06016063, 0.02226048],
                    [-0.10174976, 0.02226048, 0.10162983],
                ],
                [
                    [0.40526569, 0.18437279, 0.05891556],
                    [0.18437279, 0.13535137, 0.0603222],
                    [0.05891556, 0.0603222, 0.09712359],
                ],
                [
                    [0.13208355, 0.03362673, -0.01208926],
                    [0.03362673, 0.06261538, 0.01699577],
                    [-0.01208926, 0.01699577, 0.08031248],
                ],
                [
                    [0.13623408, 0.03036055, -0.09287403],
                    [0.03036055, 0.06499729, 0.06576895],
                    [-0.09287403, 0.06576895, 0.49017089],
                ],
            ]
        )
        pis = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        lkl = likelihood(image_matrix, means, covariances, pis, num_components)
        self.assertEqual(
            np.round(lkl),
            -46437.0,
            msg="Incorrect likelihood value returned. Make sure to use natural log",
        )
        # expected_lkl =
        print_success_message()

    def test_gmm_train(self, train_model=train_model, likelihood=likelihood):
        """Test the training
        procedure for GMM.

        returns:
        gmm = GaussianMixtureModel
        """
        image_file = "images/Starry.png"
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        num_components = 5
        m, n = image_matrix.shape

        means = np.array(
            [
                [0.34901962, 0.3647059, 0.30588236],
                [0.9882353, 0.3254902, 0.19607843],
                [1.0, 0.6117647, 0.5019608],
                [0.37254903, 0.3882353, 0.2901961],
                [0.3529412, 0.40784314, 1.0],
            ]
        )
        covariances = np.array(
            [
                [
                    [0.13715639, 0.03524152, -0.01240736],
                    [0.03524152, 0.06077217, 0.01898307],
                    [-0.01240736, 0.01898307, 0.07848206],
                ],
                [
                    [0.3929004, 0.03238055, -0.10174976],
                    [0.03238055, 0.06016063, 0.02226048],
                    [-0.10174976, 0.02226048, 0.10162983],
                ],
                [
                    [0.40526569, 0.18437279, 0.05891556],
                    [0.18437279, 0.13535137, 0.0603222],
                    [0.05891556, 0.0603222, 0.09712359],
                ],
                [
                    [0.13208355, 0.03362673, -0.01208926],
                    [0.03362673, 0.06261538, 0.01699577],
                    [-0.01208926, 0.01699577, 0.08031248],
                ],
                [
                    [0.13623408, 0.03036055, -0.09287403],
                    [0.03036055, 0.06499729, 0.06576895],
                    [-0.09287403, 0.06576895, 0.49017089],
                ],
            ]
        )
        pis = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        initial_lkl = likelihood(image_matrix, means, covariances, pis, num_components)
        MU, SIGMA, PI, r = train_model(
            image_matrix,
            num_components,
            convergence_function=default_convergence,
            initial_values=(means, covariances, pis),
        )
        final_lkl = likelihood(image_matrix, MU, SIGMA, PI, num_components)
        likelihood_difference = final_lkl - initial_lkl
        likelihood_thresh = 90000
        diff_check = likelihood_difference >= likelihood_thresh
        self.assertTrue(
            diff_check,
            msg=(
                "Model likelihood increased by less"
                " than %d for a two-mean mixture" % likelihood_thresh
            ),
        )

        print_success_message()
        print("all pass!")


if __name__ == "__main__":
    unittest.main()
