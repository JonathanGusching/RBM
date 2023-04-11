import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from keras.datasets import mnist

class RBM:
	def __init__(self, dataset="binaryalphadigs"):
		"""Initialise les données de la machine de Boltzmann.
  		Parameters:
		-----------
			dataset: la base de données que l'on veut charger pour l'apprentissage: mnist ou binaryalphadigs

		Attributes:
		-----------
			X: matrice des images de la base de données. Chaque ligne
			   correspond à une image entière applatit pour en faire un
			   vecteur 1D. Les images sont de 20*16 donc 320 pixels et il
			   existe 39 examplaires de chaque caractère. Donc X est de
			   dimension (n=39k)*(p=320)
      
			y: labels des images.
   
			erreur: historique des erreurs de prédiction de la machine.
  		"""
     
		# Lecture des données
		if(dataset == 'binaryalphadigs'):
			self.M = scipy.io.loadmat('binaryalphadigs.mat')
			self.X = self.M["dat"]
			self.y = self.M['classlabels'][0]
			self.classcounts = self.M['classcounts'][0][0][0][0]
			# Permet d'obtenir X de la forme (n=39k)*(p=320)
			arr = []
			for i in range(self.X.shape[0]):
				for j in range(self.X.shape[1]):
					arr.append(self.X[i,j].flatten())
			# Création de notre jeu de données
			self.X = np.array(arr)
			# Dimension des images pour ce jeu
			self.img_H = 20 
			self.img_W = 16

		elif(dataset == 'mnist'):
			# Permet d'obtenir les N premiers exemplaires d'un caractère
			self.classcounts = 39
			self.mnist_number_examples = 6000
			(self.X, self.y), (_, _) = mnist.load_data()
			rearrange = np.argsort(self.y)
			self.X = (self.X[rearrange] > 0.5) * 1
			self.y = (self.y[rearrange] > 0.5) * 1
			temp = []
			for i in range(10):
				temp.append(self.X[i*self.mnist_number_examples:i*self.mnist_number_examples + self.classcounts].reshape((self.classcounts,28*28)))
			# Création de notre jeu de données
			self.X = np.concatenate(temp)
			# Dimension des images pour ce jeu
			self.img_H = 28 
			self.img_W = 28

		self.erreur = []
  
	def print_info(self):
		"""Afficher en l'ensemble des informations qui définissent la machine
		   de Boltzmann.
  		"""
     
		print("X: ", self.X)
		print("y: ", self.y)
		print("W: ", self.W)
		print("a: ", self.a)
		print("b: ", self.b)

		print("X shape: ", self.X.shape)
		print("y shape: ", self.y.shape)
		print("W shape: ", self.W.shape)
		print("a shape: ", self.a.shape)
		print("b shape: ", self.b.shape)
        
	def plot_images(self, images):
		"""Affiche un nombre ``num_images`` d'images prises 
  		   aléatoirement dans les données.
  		"""
		
		num_images = 50
		
		fig = plt.figure(figsize=(10, 10))
		rows = 5
		columns = 10
		
		indices = np.random.choice(np.arange(images.shape[0]), num_images, replace=False)
		
		i = 1
		for indice in indices:
			fig.add_subplot(rows, columns, i)
			plt.imshow(images[indice].reshape(self.img_H, self.img_W), cmap="Greys")
			i += 1
		plt.show()

	def lire_alpha_digit(self, cible):
		"""Permet de récupérer les données sous forme matricielle.
		
		Parameters:
		-----------
			cible: liste des caractères que l'on souhaite apprendre
			
			
		Returns:
		--------
			alpha_digit_data: matrice des données sélectionnées
		"""
		
		nombre_variete = self.classcounts
		alpha_digit_data = np.zeros((nombre_variete * len(cible), self.p))
		
		index = 0
		for cible in cible:
			indice_cible = (cible*nombre_variete, (cible+1)*nombre_variete)
			alpha_digit_data[index * nombre_variete:(index + 1) * nombre_variete] = self.X[indice_cible[0]:indice_cible[1]]
			index += 1
			
		return alpha_digit_data

	def init_RBM(self, p, q):
		"""Initialise les poids et les biais de la RBM
  
		Parameters:
		-----------
			p: dimension des images applaties.
   
			q: nombre de couches cachées de la machine.
		
		Returns:
		--------
			obj: structure RBM avec des poids et des biais initialisés.
		"""
		
		self.p = p
		self.q = q
		
		std = np.sqrt(0.01)
		
		self.W = np.random.normal(0, std, size=(p, q))
		self.a = np.zeros(p)
		self.b = np.zeros(q)
		
		return self

	def entree_sortie_RBM(self, X):
		"""Calcul du forward: sigmoide sur les données d'entrée.
  
		Parameters:
		-----------
			X: ensemble des images des caractères que l'on souhaite apprendre.
   
		Returns:
		--------
			forward: np.array_like contenant la sigmoid des données d'entrée
					 multipliées par les poids de la machine et sommées avec un biais.
  		"""
    
		return 1/ (1 + np.exp(-( X @ self.W + self.b)))

	def sortie_entree_RBM(self, H):
		"""Calcul du backward: sigmoide sur les données de sortie.
  
		Parameters:
		-----------
			H: matrice des couches cachées de la machine.
   
		Returns:
		--------
			backward: np.array_like contenant la sigmoid des données de sortie
					  multipliées par les poids de la machine et sommées avec un biais.
  		"""
    
		return 1 / (1 + np.exp(-(H @ self.W.T + self.a)))

	def train_RBM(self, taille_batch, lr, epochs, caractere_liste = [5, 6, 7] ):
		"""Entrainement de la machine.
  
		Parameters:
		-----------
			taille_batch: taille des batchs à considérer pour l'entrainement.
   
			lr: learning_rate à considérer pour l'entrainement.

			epochs: nombre d'époque à considérer pour l'entrainement.

			caractere_liste: la liste de caractères sur laquelle on entraîne le modèle
  		"""
	
		# Sélection des caractères devant être appris
		caractere_apprendre = self.lire_alpha_digit(caractere_liste)

		# on garde les caractères juste pour l'affichage
		self.caracteres = caractere_liste
		
		for _ in range(epochs):
			caractere_apprendre = np.random.permutation(caractere_apprendre)
			indice = 0
			# Parcours des batchs
			while indice  * taille_batch < caractere_apprendre.shape[0]:
				X_batch = caractere_apprendre[indice * taille_batch:(indice + 1 )*taille_batch, :]
				tb = X_batch.shape[0]
    
				# Calcul des distributions
				V_0 = X_batch
				p_h_v_0 = self.entree_sortie_RBM(V_0)
				h_0 = (np.random.rand(tb, self.q) < p_h_v_0) * 1
				p_v_h_0 = self.sortie_entree_RBM(h_0)
				V_1 = (np.random.rand(tb, self.p) < p_v_h_0) * 1
				p_h_v_1 = self.entree_sortie_RBM(V_1)

				# Calcul des gradients
				grad_a = np.sum(V_0 - V_1, axis = 0)
				grad_b = np.sum(p_h_v_0 - p_h_v_1, axis = 0)
				grad_W = V_0.T @ p_h_v_0 - V_1.T @ p_h_v_1

				# Mis-à-jour des paramètres de la machine
				self.W = self.W + lr*grad_W/tb
				self.a = self.a + lr*grad_a/tb
				self.b = self.b + lr*grad_b/tb
    
				indice += 1
    
			# Création des couche cachées de la machine
			H = self.entree_sortie_RBM(caractere_apprendre)
   
			# Reconstruction de X
			X_rec = self.sortie_entree_RBM(H)
   
			# Estimation de l'erreur de reconstruction
			self.erreur.append(np.sum((caractere_apprendre - X_rec)**2)/caractere_apprendre.shape[0])
   
	def plot_erreur(self, stacked=False, label='q'):
		"""
		Affiche la courbe de l'historique de l'erreur de prédiction de 
		   la machine pendant l'entrainement.

		Parameters:
		-----------
			stacked: ligne qui permet d'enchaîner les affichages d'erreur sans avoir le plt.show()

			label: si l'on choisit d'afficher en légendes les caractères 'c' ou le nombre d'unités cachées 'q'
  		"""
		
		if len(self.erreur) != 0:
			if(label == 'c'):
				plt.plot(self.erreur, label='c=' + str(self.caracteres))
			else:
				plt.plot(self.erreur, label='q=' + str(self.q))
			if(not stacked):
				plt.show()
		else:
			print("Aucun entrainement n'a encore été effectué.")

	def generer_image_RBM(self, n_donnees, n_iter_gibbs):
		"""Génére un nombre ``n_donnees`` d'images à partir de la machine 
		   de Boltzmann entrainée via un échantiollonage de Gibbs.
     
		Parameters:
		-----------
			n_donnees: nombre d'image à générer par la machine.
   
			n_iter_gibbs: nombre d'itération pour pour l'échantillonage de Gibbs.
   
		Returns:
		--------
			images: np.array_like de dimension (n=39*n_donnee)*(p=320) représentant
				    les images générées.
  		"""
    
		images = []
		for _ in range(n_donnees):
			v = (np.random.rand(self.p) < 1/2) * 1
			for _ in range(n_iter_gibbs):
				h = (np.random.rand(self.q) < self.entree_sortie_RBM(v)) * 1
				v = (np.random.rand(self.p) < self.sortie_entree_RBM(h)) * 1
			images.append(v)
   
		return np.array(images)

if __name__ == "__main__":
	CHOICE = 6
	
	if(CHOICE == 0):
		plt.rcParams.update({'font.size': 20})
		for q in [10, 50, 100, 250, 500, 1000]:
			rbm = RBM().init_RBM(320, q)
			rbm.train_RBM(64, 0.01, 1000)
			rbm.plot_erreur(True)
			generated_images = rbm.generer_image_RBM(50, 1000)
		plt.title("MSE sur les images générées")
		plt.ylabel("erreur")
		plt.xlabel("epoch")
		plt.grid()
		plt.legend()
		plt.show()
	elif(CHOICE == 1):
		plt.rcParams.update({'font.size': 12})
		for q in [10, 1000]:
			rbm = RBM().init_RBM(320, q)
			rbm.plot_images(rbm.X)
			rbm.train_RBM(64, 0.01, 1000)
			generated_images = rbm.generer_image_RBM(50, 1000)
			rbm.plot_images(generated_images)
	elif(CHOICE == 2):
		plt.rcParams.update({'font.size': 20})
		for c in range(36):
			rbm = RBM().init_RBM(320, 100)
			rbm.train_RBM(64, 0.01, 500, caractere_liste = [i for i in range(c+1)])
			rbm.plot_erreur(True, label='c')
			generated_images = rbm.generer_image_RBM(50, 1000)
		plt.title("MSE sur les images générées")
		plt.ylabel("erreur")
		plt.xlabel("epoch")
		plt.grid()
		#plt.legend()
		plt.show()
	#MNIST
	elif(CHOICE==3):
		plt.rcParams.update({'font.size': 20})
		for q in [10, 50, 100, 250, 500, 1000]:
			rbm = RBM(dataset='mnist').init_RBM(28*28, q)
			rbm.train_RBM(64, 0.01, 1000)
			rbm.plot_erreur(True)
			generated_images = rbm.generer_image_RBM(50, 1000)
		plt.title("MSE sur les images générées")
		plt.ylabel("erreur")
		plt.xlabel("epoch")
		plt.grid()
		plt.legend()
		plt.show()
	elif(CHOICE == 4):
		plt.rcParams.update({'font.size': 12})
		for q in [10, 500]:
			rbm = RBM(dataset='mnist').init_RBM(28*28, q)
			rbm.plot_images(rbm.X)
			rbm.train_RBM(64, 0.01, 1000)
			generated_images = rbm.generer_image_RBM(50, 1000)
			rbm.plot_images(generated_images)

	elif(CHOICE == 5):
		plt.rcParams.update({'font.size': 12})
		for n_iter_gibbs in (10, 25, 50, 100, 250, 500, 750, 1000):
			rbm = RBM().init_RBM(320, 250)
			#rbm.plot_images(rbm.X)
			rbm.train_RBM(64, 0.01, 1000, caractere_liste = [i for i in range(36)])
			generated_images = rbm.generer_image_RBM(50, n_iter_gibbs)
			rbm.plot_images(generated_images)

	elif(CHOICE == 6):
		plt.rcParams.update({'font.size': 12})
		for n_epochs in [50, 100, 250, 500, 1000]:
			rbm = RBM().init_RBM(320, 250)
			#rbm.plot_images(rbm.X)
			rbm.train_RBM(64, 0.01, n_epochs, caractere_liste = [i for i in range(36)])
			generated_images = rbm.generer_image_RBM(50, 500)
			rbm.plot_images(generated_images)