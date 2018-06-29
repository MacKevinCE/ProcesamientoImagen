import sys
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy import misc


# Redução através de operadores de produção local (mediana)
def reduccionResolucion(img, n, m):
	# Tamanho de imagem reduzido
	N = int(img.shape[0] / n)
	M = int(img.shape[1] / m)
	if (N > 0 and M > 0):
		if (n == 1 and m == 1):
			return img
		else:
			imgout = np.zeros([N, M])
			for i in np.arange(N):
				# Tamanho do bloco em x
				rx = [i * n, min((i + 1) * n, img.shape[0])]
				for j in np.arange(M):
					# Tamanho do bloco em y
					ry = [j * m, min((j + 1) * m, img.shape[1])]
					# Bloco da imagem 'img' de um tamanho aproximado de 'n*m'
					bloque = img[rx[0]:rx[1], ry[0]:ry[1]]
					# Pixel da imagem reduzida é a mediana do bloco
					imgout[i][j] = np.median(bloque)
			return np.uint8(imgout)
	else:
		return None


# Função cúbica
def cubica(p0, p1, p2, p3, x):
	p0 = float(p0)
	p1 = float(p1)
	p2 = float(p2)
	p3 = float(p3)
	return float(p1 + 0.5 * x * (p2 - p0 + x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + x * (3.0 * (p1 - p2) + p3 - p0))))


# Algoritmo de interpolação bicúbica
def ampliacionResolucion(img, n, m):
	# Tamanho das imagens
	N = img.shape[0]
	M = img.shape[1]
	if ((N > 0 and M > 0) or (n == 1 and m == 1)):
		# Tamanho de imagem ampliado
		N = N * n
		M = M * m

		# Envolve a imagem com 1 linha para a direita e 2 para a esquerda
		img = np.concatenate([[img[0, :]], img, [img[-1, :]], [img[-1, :]]], axis=0)

		# Envolve a imagem com 1 linha para cima e 2 para baixo
		img = np.concatenate([np.transpose([img[:, 0]]), img, np.transpose([img[:, -1], img[:, -1]])], axis=1)

		imgout = np.zeros([N, M])
		for i in np.arange(N):
			for j in np.arange(M):
				# Obter os índices da imagem original e os decimais
				antX = i / n
				difX = antX - int(antX)
				antX = int(antX)

				antY = j / m
				difY = antY - int(antY)
				antY = int(antY)

				# Em p obter o cubica() das linhas de um bloco 4 * 4 e difX
				p = np.zeros(4)
				for k in np.arange(4):
					p[k] = cubica(img[antX + k][antY],
								  img[antX + k][antY + 1],
								  img[antX + k][antY + 2],
								  img[antX + k][antY + 3], difX)

				# Obtém o cubica() de p e difY
				imgout[i][j] = int(cubica(p[0], p[1], p[2], p[3], difY))
		return np.uint8(imgout)
	else:
		return img


def interpolacion(img, mascara):
	# Tamanho das imagens
	N = min(img.shape[0], mascara.shape[0])
	M = min(img.shape[1], mascara.shape[1])

	# é equivalente a 'a > b'
	# Qualquer pixel maior que 0 resultará em 'True' para cada pixel de mascara
	Mascara = mascara.__gt__(0)

	for i in np.arange(N):
		for j in np.arange(M):
			# Lista de pixels resultantes da interpolação
			pixel = []

			# 'True' se o pixel é desconhecida
			if (Mascara[i][j]):
				# Interpolação à esquerda
				if (j >= 2):
					if (mascara[i][j - 1] == 0 and mascara[i][j - 2] == 0):
						pixel.append(2 * img[i][j - 1] - img[i][j - 2])

				# Interpolação à direita
				if (j < M - 2):
					if (mascara[i][j + 1] == 0 and mascara[i][j + 2] == 0):
						pixel.append(2 * img[i][j + 1] - img[i][j + 2])

				# Interpolação ascendente
				if (i >= 2):
					if (mascara[i - 1][j] == 0 and mascara[i - 2][j] == 0):
						pixel.append(2 * img[i - 1][j] - img[i - 2][j])

				# Interpolação descendente
				if (i < N - 2):
					if (mascara[i + 1][j] == 0 and mascara[i + 2][j] == 0):
						pixel.append(2 * img[i + 1][j] - img[i + 2][j])

				# Interpolação esquina ascendente esquerda
				if (i >= 2 and j >= 2):
					if (mascara[i - 1][j - 1] == 0 and mascara[i - 2][j - 2] == 0):
						pixel.append(2 * img[i - 1][j - 1] - img[i - 2][j - 2])

				# Interpolação esquina descendente esquerda
				if (i < N - 2 and j >= 2):
					if (mascara[i + 1][j - 1] == 0 and mascara[i + 2][j - 2] == 0):
						pixel.append(2 * img[i + 1][j - 1] - img[i + 2][j - 2])

				# Interpolação esquina ascendente direita
				if (i >= 2 and j < M - 2):
					if (mascara[i - 1][j + 1] == 0 and mascara[i - 2][j + 2] == 0):
						pixel.append(2 * img[i - 1][j + 1] - img[i - 2][j + 2])

				# Interpolação esquina descendente direita
				if (i < N - 2 and j < M - 2):
					if (mascara[i + 1][j + 1] == 0 and mascara[i + 2][j + 2] == 0):
						pixel.append(2 * img[i + 1][j + 1] - img[i + 2][j + 2])

				# Interpolação bem sucedida pelo menos um
				if (len(pixel) > 0):
					pixel = np.array(pixel)
					# Pixel desconhecido é igual à mediana das interpolações
					img[i][j] = np.uint8(np.median(pixel))
					# Não é mais um pixel desconhecido
					Mascara[i][j] = False
	# Retorna a imagem e a mascara com os pixels encontrados em preto (0)
	return img, mascara * Mascara


# Algoritmo da pirâmide de Gaussiana
def piramideGaussiana(img, mascara, ruta, debug=False):
	# Lista para salvar as imagens que vamos obter
	imgP = []
	mascaraP = []
	pixelesDesconocidos = True

	# Enquanto encontra pixels desconhecidos
	while (pixelesDesconocidos):
		pixelesDesconocidos = False
		if (np.sum(mascara.__gt__(0)) > 0 and img.shape[0] > 1 and mascara.shape[1] > 1):
			pixelesDesconocidos = True

			# Salvar imagens na lista
			imgP.append(img)
			mascaraP.append(mascara)

			# Obter a imagem com os novos pixels encontrados pela interpolação e a nova máscara
			img, mascara = interpolacion(img, mascara)
			# Reduzir a imagem e a mascara para mitade de largura e longo
			# img = misc.imresize(img, 50, interp="bicubic")
			img = reduccionResolucion(img, 2, 2)
			# mascara = misc.imresize(mascara, 50, interp="bicubic")
			mascara = reduccionResolucion(mascara, 2, 2)

			if (debug):
				# Mostrando as imagens obtidas após a redução e interpolação
				plt.subplot(1, 2, 1)
				plt.imshow(img, cmap=plt.cm.gray)
				plt.title("Fotografia " + str(img.shape[0]) + " X " + str(img.shape[1]))
				plt.subplot(1, 2, 2)
				plt.imshow(mascara, cmap=plt.cm.gray)
				plt.title("Mascara " + str(mascara.shape[0]) + " X " + str(mascara.shape[1]))

				plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05, right=0.99)
				plt.tight_layout()
				plt.savefig(ruta + "/interpolacion_reduccion_" + str(len(imgP)) + ".png")
	# Enquanto não estamos no topo da pirâmide
	while (len(imgP) != 0):
		# Ampliar a imagem img
		img = ampliacionResolucion(img, 2, 2)

		# Obter as imagens salvas na lista
		imgA = imgP.pop()
		mascaraA = mascaraP.pop()

		if (debug):
			plt.subplot(1, 2, 1)
			plt.imshow(img, cmap=plt.cm.gray)
			plt.title("Antes " + str(imgA.shape[0]) + " X " + str(imgA.shape[1]))

		# Mostrando a imagen obtida antes de copiar os novos pixels
		N = min(imgA.shape[0], img.shape[0])
		M = min(imgA.shape[1], img.shape[1])
		for i in np.arange(N):
			for j in np.arange(M):
				# Se a máscara indica o pixel desconhecido fazer
				if (mascaraA[i][j] != 0):
					# Copiar os pixels conhecidos da imagem ampliada para a imagem com pixels desconhecidos
					imgA[i][j] = img[i][j]
		if (debug):
			# Mostrando a imagen obtida depois de copiar os novos pixels
			plt.subplot(1, 2, 2)
			plt.imshow(imgA, cmap=plt.cm.gray)
			plt.title("Depois " + str(imgA.shape[0]) + " X " + str(imgA.shape[1]))

			plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05, right=0.99)
			plt.tight_layout()
			plt.savefig(ruta + "/ampliacion_copia_" + str(len(imgP)) + ".png")

		# Substituir a nova imagem pela qual possui os pixels conhecidos
		img = imgA
	return img


def histograma(img, nbins):
	N = img.shape[0]
	M = img.shape[1]

	hist = np.zeros(nbins).astype(int)
	for i in np.arange(N):
		for j in np.arange(M):
			hist[img[i][j]] += 1
	return hist


def histogram_equalization(img, values):  # equaliza las imagenes
	N = img.shape[0]
	M = img.shape[1]

	hist = histograma(img, values) * (values - 1) / float(N * M)

	imgout = np.array(img)
	imgout[np.where(img == 0)] = hist[0]
	for i in np.arange(1, values):
		hist[i] = hist[i] + hist[i - 1]  # histograma acumulado
		imgout[np.where(img == i)] = hist[i]

	return imgout.astype("uint8")


def error(original, img):
	N = original.shape[0]
	M = img.shape[1]

	original = original.astype(float)
	img = img.astype(float)

	return np.sqrt(np.sum((original - img) ** 2) / (N * M))


def dominioEspacio(img, F):
	tam = img.shape
	F = F[:3, :3]
	r = np.array([[-1, 2], [-1, 2]]) + 1

	# encerra a matriz com uma linha e coluna de zeros ao redor
	Aux = np.concatenate([np.zeros([1, img.shape[1]]), img, np.zeros([1, img.shape[1]])], axis=0)
	Aux = np.concatenate([np.zeros([Aux.shape[0], 1]), Aux, np.zeros([Aux.shape[0], 1])], axis=1)

	Resul = np.zeros(img.shape)
	for i in np.arange(tam[0]):
		for j in np.arange(tam[1]):
			# a multiplicação ponto-a-ponto do corte da matriz Aux com F é adicionada
			Resul[i][j] = np.sum(Aux[r[0][0] + i:r[0][1] + i, r[1][0] + j:r[1][1] + j] * F)

	return Resul


def crearMascara(original, img, dif):
	original = original.astype(int)
	img = img.astype(int)
	return (np.abs(original - img).__gt__(dif) * 255).astype("uint8")


def main():
	# Arquivo de leitura de input
	sys.stdin = open('debug_original.in', 'r')

	debug = bool(int(input()))
	isMascara = bool(int(input()))
	isOriginal = bool(int(input()))
	isEqualization = bool(int(input()))
	isFiltro = bool(int(input()))

	# Path no qual as imagens obtidas serão salvas
	ruta = str(input()).rstrip()

	# Fotografia a ser restaurada
	foto = imageio.imread(str(input()).rstrip())

	# Máscara da fotografia a ser restaurada
	if (isMascara):
		mascara = imageio.imread(str(input()).rstrip())

	# Fotografia original para depuração e outros
	if (isOriginal):
		original = imageio.imread(str(input()).rstrip())

	# Algoritmo para criar uma máscara a partir da fotografia e do original
	if (isMascara == False and isOriginal):
		mascara = crearMascara(original, foto, 0)
		imageio.imwrite(ruta + "/mascara_creada.png", mascara)

	# Debug
	if (debug):
		sys.stdout = open(ruta + "/salida.out", 'w')
		plt.subplot(111)
		plt.imshow(foto, cmap=plt.cm.gray)
		plt.title("Fotografia danificada")

		plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05, right=0.99)
		plt.tight_layout()
		plt.savefig(ruta + "/_foto.png")

		plt.subplot(111)
		plt.imshow(mascara, cmap=plt.cm.gray)
		plt.title("Máscara de fotografia")

		plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05, right=0.99)
		plt.tight_layout()
		plt.savefig(ruta + "/_mascara.png")

		plt.subplot(111)
		plt.imshow(original, cmap=plt.cm.gray)
		plt.title("Fotografia original")

		plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05, right=0.99)
		plt.tight_layout()
		plt.savefig(ruta + "/_original.png")

		plt.subplot(121)
		plt.imshow(foto, cmap=plt.cm.gray)
		plt.title("Fotografia danificada")

		plt.subplot(122)
		plt.imshow(mascara, cmap=plt.cm.gray)
		plt.title("Máscara de fotografia")

		plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05, right=0.99)
		plt.tight_layout()
		plt.savefig(ruta + "/foto_mascara.png")

		if (isOriginal):
			print("MRSE(original,foto):")
			MRSE1 = error(original, foto)
			print(MRSE1)

	# Algoritmo de pirâmide gaussiana
	if (isOriginal or isMascara):
		foto = piramideGaussiana(foto, mascara, ruta, debug)
		imageio.imwrite(ruta + "/restaurada.png", foto)

	if (debug):
		if (isOriginal):
			print("MRSE(original,restaurada):")
			MRSE2 = error(original, foto)
			print(MRSE2)

			print("Mejora:" + str(MRSE1 - MRSE2))
			print("Mejora:" + str((MRSE1 - MRSE2) / MRSE1 * 100) + "%")

		plt.subplot(111)
		plt.imshow(foto, cmap=plt.cm.gray)
		plt.title("Fotografia restaurada")

		plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05, right=0.99)
		plt.tight_layout()
		plt.savefig(ruta + "/_restaurada.png")

	# Equalizar imagem
	if (isEqualization):
		foto = histogram_equalization(foto, values=256)
		imageio.imwrite(ruta + "/ecualizada.png", foto)

	# Filtro
	if (isFiltro):
		f = np.ones([3, 3]) / 9
		foto = dominioEspacio(foto, f).astype("uint8")
		imageio.imwrite(ruta + "/filtro.png", foto)


if __name__ == "__main__":
	main()
