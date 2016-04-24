import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy
import skimage.io as io
import collections



def ComplementStainMatrix(W):
    WComp = W

    # calculate directed cross-product of first two columns
    if (W[0, 0]**2 + W[0, 1]**2) > 1:
        WComp[0, 2] = 0
    else:
        WComp[0, 2] = (1 - (W[0, 0]**2 + W[0, 1]**2))**0.5

    if (W[1, 0]**2 + W[1, 1]**2) > 1:
        WComp[1, 2] = 0
    else:
        WComp[1, 2] = (1 - (W[1, 0]**2 + W[1, 1]**2))**0.5

    if (W[2, 0]**2 + W[2, 1]**2) > 1:
        WComp[2, 2] = 0
    else:
        WComp[2, 2] = (1 - (W[2, 0]**2 + W[2, 1]**2))**0.5

    # normalize new vector to unit-norm
    WComp[:, 2] = WComp[:, 2] / numpy.linalg.norm(WComp[:, 2])

    return WComp

def ColorDeconvolution(I, W):

    # complement stain matrix if needed
    if numpy.linalg.norm(W[:, 2]) <= 1e-16:
        Wc = ComplementStainMatrix(W)
    else:
        Wc = W.copy()

    # normalize stains to unit-norm
    for i in range(Wc.shape[1]):
        Norm = numpy.linalg.norm(Wc[:, i])
        if Norm >= 1e-16:
            Wc[:, i] /= Norm

    # invert stain matrix
    Q = numpy.linalg.inv(Wc)

    # transform 3D input image to 2D RGB matrix format
    m = I.shape[0]
    n = I.shape[1]
    if I.shape[2] == 4:
        I = I[:, :, (0, 1, 2)]
    I = numpy.reshape(I, (m * n, 3))

    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    I = I.astype(dtype=numpy.float32)
    I[I == 0] = 1e-16
    ODfwd = -(255 * numpy.log(I / 255)) / numpy.log(255)
    ODdeconv = numpy.dot(ODfwd, numpy.transpose(Q))
    ODinv = numpy.exp(-(ODdeconv - 255) * numpy.log(255) / 255)

    # reshape output
    StainsFloat = numpy.reshape(ODinv, (m, n, 3))

    # transform type
    Stains = numpy.copy(StainsFloat)
    Stains[Stains > 255] = 255
    Stains = Stains.astype(numpy.uint8)

    Unmixed = collections.namedtuple('Unmixed', ['Stains', 'StainsFloat', 'Wc'])
    Output = Unmixed(Stains, StainsFloat, Wc)
    return Output


def ColorConvolution(I, W):

    # transform 3D input image to 2D RGB matrix format
    m = I.shape[0]
    n = I.shape[1]
    I = numpy.reshape(I, (m * n, 3))

    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    I = I.astype(dtype=numpy.float32)
    ODfwd = -(255 * numpy.log(I / 255)) / numpy.log(255)
    ODdeconv = numpy.dot(ODfwd, numpy.transpose(W))
    ODinv = numpy.exp(-(ODdeconv - 255) * numpy.log(255) / 255)

    # reshape output
    IOut = numpy.reshape(ODinv, (m, n, 3))
    IOut[IOut > 255] = 255
    IOut = IOut.astype(numpy.uint8)
    return IOut


# open image
I = io.imread('breast01.jpg')

# Define input image and H&E color matrix
W = numpy.array([[0.650, 0.072, 0.268],
                 [0.704, 0.990, 0.570],
                 [0.286, 0.105, 0.776]])


# perform color deconvolution
Unmixed = ColorDeconvolution(I, W)

# reconvolve color image
Remixed = ColorConvolution(Unmixed.Stains, W)

# color image of hematoxylin stain
WH = W.copy()
WH[:, 1] = 0
WH[:, 2] = 0
Hematoxylin = ColorConvolution(Unmixed.Stains, WH)



# color image of eosin stain
WE = W.copy()
WE[:, 0] = 0
WE[:, 2] = 0
Eosin = ColorConvolution(Unmixed.Stains, WE)

WD = W.copy()
WD[:, 0] = 0
WD[:, 1] = 0
DAB = ColorConvolution(Unmixed.Stains, WD)
io.imsave('DAB.jpg', DAB)

# view output - check if coherent
plt.subplot(2, 2, 1)
plt.imshow(I)
plt.title('Input image')
plt.subplot(2, 2, 3)
plt.imshow(Hematoxylin)
plt.title('Hematoxylin')
plt.subplot(2, 2, 4)
plt.imshow(DAB)
plt.title('DAB')

plt.show()
