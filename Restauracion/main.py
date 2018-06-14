import numpy
import imageio
import scipy.stats as stats
import sys


# sys.stdin = open('case1.in', 'r')

def HistogramaCumulativo(L):
	ha = numpy.histogram(L, range(257), density=True)[0]  # histograma normalizado
	for i in range(1, 256):
		ha[i] = ha[i] + ha[i - 1]  # histograma acumulado
	return ha * 255.0


def equalizar(ha, L):  # equaliza las imagenes
	N = L.shape[0]
	M = L.shape[1]
	for i in range(N):
		for j in range(M):
			L[i][j] = ha[L[i][j]]
	return L


def operador(img, type=0):
	if (type == 1):
		pixel = numpy.median(img)
	elif (type == 2):
		pixel = stats.gmean(img)
	else:
		pixel = img.mean()

	return pixel


def reduccionResolucion(img, n, m):
	N = int(img.shape[0] / n)
	M = int(img.shape[1] / m)
	imgout = numpy.zeros([N, M])
	for i in numpy.arange(N):
		rx = [i * n, min((i + 1) * n, img.shape[0])]
		for j in numpy.arange(M):
			ry = [j * m, min((j + 1) * m, img.shape[1])]
			bloque = img[rx[0]:rx[1], ry[0]:ry[1]]
			imgout[i][j] = operador(bloque, 2)
	return numpy.uint8(imgout)


def ampliacionResolucion(img, n, m):
	N = img.shape[0]
	M = img.shape[1]
	imgo = img

	for i in numpy.arange(int(n / 2)):
		img = numpy.concatenate([[img[i * 2, :]], img, [img[-i * 2, :]]], axis=0)
	for i in numpy.arange(int(m / 2)):
		img = numpy.concatenate([numpy.transpose([img[:, i * 2]]), img, numpy.transpose([img[:, -i * 2]])], axis=1)
	img = img / 255

	imgout = numpy.zeros([N * n, M * m])
	for i in numpy.arange(N):
		for j in numpy.arange(M):
			V = img[i:i + n, j:j + m]
			W = (V.max() - V.min()) * numpy.std(V)
			V = img[i][j] + V * W
			imgout[i * n:(i + 1) * n, j * m:(j + 1) * m] = V
	return numpy.uint8(imgout * 255)


def interpolacion(img, mascara):
	N = min(img.shape[0], mascara.shape[0])
	M = min(img.shape[1], mascara.shape[1])
	Mascara = mascara.__gt__(0)
	for i in numpy.arange(N):
		for j in numpy.arange(M):
			pixel = []
			if (mascara[i][j] > 0):
				if (j >= 2):
					if (mascara[i][j - 1] == 0 and mascara[i][j - 2] == 0):
						pixel.append(2 * img[i][j - 1] - img[i][j - 2])
				if (j < M - 2):
					if (mascara[i][j + 1] == 0 and mascara[i][j + 2] == 0):
						pixel.append(2 * img[i][j + 1] - img[i][j + 2])
				if (i >= 2):
					if (mascara[i - 1][j] == 0 and mascara[i - 2][j] == 0):
						pixel.append(2 * img[i - 1][j] - img[i - 2][j])
				if (i < N - 2):
					if (mascara[i + 1][j] == 0 and mascara[i + 2][j] == 0):
						pixel.append(2 * img[i + 1][j] - img[i + 2][j])
				if (len(pixel) > 0):
					pixel = numpy.array(pixel)
					img[i][j] = numpy.uint8(numpy.median(pixel))
					Mascara[i][j] = False
	return img, mascara * Mascara


def piramideGaussiana(img, mascara):
	imgP = []
	mascaraP = []
	desconocido = True
	while (desconocido):
		desconocido = False
		if (numpy.sum(mascara.__gt__(0)) > 0):
			imgP.append(img)
			mascaraP.append(mascara)
			desconocido = True
			img, mascara = interpolacion(img, mascara)
			img = reduccionResolucion(img, 2, 2)
			mascara = reduccionResolucion(mascara, 2, 2)

	while (len(imgP) != 0):
		img = ampliacionResolucion(img, 2, 2)
		imgA = imgP.pop()
		mascaraA = mascaraP.pop()
		N = min(imgA.shape[0], img.shape[0])
		M = min(imgA.shape[1], img.shape[1])
		for i in numpy.arange(N):
			for j in numpy.arange(M):
				if (mascaraA[i][j] != 0):
					imgA[i][j] = img[i][j]
		img = imgA

	return img


def error(original, img):
	n = original.shape[0]
	m = img.shape[1]
	return numpy.sqrt(numpy.sum((original - img) ** 2) / (n * m))


def renormalizar(g):
	maximo = numpy.sum(g)
	if maximo > 0:  # quando é igual a zero não é normalizado
		g = (g * 255) / maximo  # regra simples de três
	return numpy.uint8(numpy.real(g))  # convierte g en Unsigned integer (0 to 255) 8 bits


def modifiar(original, img, mascara):
	print(original.shape)
	original = original[:, :, 0]
	imageio.imwrite("./original.png", original)

	print(img.shape)
	img = img[:, :, 0]
	imageio.imwrite("./foto.png", img)

	print(mascara.shape)
	mascara = mascara[:, :, 0]
	imageio.imwrite("./mascara01.png", mascara)


def main():
	nFoto = "foto01"
	nMascara = "01"
	sys.stdout = open(nFoto + "/geo-RMSE" + nMascara + ".out", 'w')
	original = imageio.imread(nFoto + "/original.png")
	img = imageio.imread(nFoto + "/foto.png")
	mascara = imageio.imread(nFoto + "/mascara" + nMascara + ".png")
	# modifiar(original,img,mascara)
	print("MRSE(original,foto):")
	MRSE1 = error(original, img)
	print(MRSE1)
	imgout = piramideGaussiana(img, mascara)
	imageio.imwrite("./" + nFoto + "/geo-restaurada" + nMascara + ".png", imgout)
	print("MRSE(original,restaurada):")
	MRSE2 = error(original, imgout)
	print(MRSE2)
	print("Mejora:")
	print(MRSE1 - MRSE2)


if __name__ == "__main__":
	main()
