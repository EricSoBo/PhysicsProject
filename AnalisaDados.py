import numpy as np
from scipy import optimize, interpolate
import matplotlib.pyplot as plt

def MinQuadrados(Dados):
  
  n = len(Dados)
  X = 0
  Y = 0
  XX = 0
  XY = 0
  for i in range(len(Dados)):
    X += Dados[i][0]
    Y += Dados[i][1]
    XX += Dados[i][0]**2
    XY += Dados[i][0]*Dados[i][1]
  
  A0 = (X*Y - n*XY) / ((X**2) - n*XX)
  A1 = (Y*XX - X*XY) / (n*XX - (X**2))

  return A0, A1

def fd_diff(xnovo, int):

	h = xnovo[1]-xnovo[0]
	f = []
	for x in range(len(xnovo)-1):
		f.append((int[x+1]-int[x])/h)
	
	return f

def cd5_diff(x, f):

	h = x[1]-x[0]
	f_ = []
	for y in range(2, len(x)-2):
		f_.append( 1/(12*h)*(f[y-2]-8*f[y-1]+8*f[y+1]-f[y+2]) )

	return f_

def HarmonicOscAmort(X, Y, Time): # Movimento Harmônico Amortecido

	Time = Time/1000 # converte para segundos
	plt.rcParams['figure.figsize'] = [11, 7]
	plt.title('Trajetória do objeto')
	plt.xlabel('Tempo (s)')
	plt.ylabel('Posição X (cm)')
	Y_mean = [np.mean(Y)]*len(Time)
	Y = Y - Y_mean
	Y_mean = [0]*len(Time)
	Y = Y*0.026458333333333 # Conversão de pixel para centímetros
	
	coef = interpolate.splrep(Time, Y, s=0) # Interpolação por Cubic-Splines
	xnovo = np.linspace(Time[0], Time[-1], num=1000, endpoint=True)
	pol_position = interpolate.splev(xnovo, coef, der=0)
	
	plt.scatter(Time, Y, color='r', label='Pontos')
	plt.plot(Time, Y_mean, 'b', label='Equilibrio')
	plt.plot(xnovo, pol_position, color='g', label='Posição por Cubic-Splines')
	plt.legend()
	plt.grid()
	plt.show()

	plt.title('Trajetória do objeto')
	plt.xlabel('Tempo (s)')
	plt.ylabel('Posição X (cm)')
	velocity = fd_diff(Time, Y)
	velocity_ = cd5_diff(Time, Y)
	coef_ = interpolate.splrep(Time[0:len(Time)-1], velocity, s=0)
	pol_velocity = interpolate.splev(xnovo, coef_, der=0)
	plt.plot(xnovo, pol_position, color='r', label='Posição por Cubic-Splines')
	plt.scatter(Time[2:len(Time)-2], velocity_, color='black', label='Velocidade')
	plt.legend()
	plt.grid()
	plt.show()

def ProjectileMotion(X, Y, Time): # Movimento de projétil

	Time = Time/1000
	plt.rcParams['figure.figsize'] = [11, 7]
	plt.title('Trajetória do objeto')
	plt.xlabel('Posição X (px)')
	plt.ylabel('Posição Y (px)')
	X = X-X[0] # Fazendo o gráfico começar na origem
	Y = Y-Y[0]
	# Como nas coordenadas do computador, o Y aumenta no sentido contrário do convencional, estou invertendo o eixo
	Y = Y*-1

	plt.scatter(X, Y)

	#coef = interpolate.splrep(Time, Y, s=0) # Interpolação por Cubic-Splines
	#xnovo = np.linspace(Time[0], Time[-1], num=1000, endpoint=True)
	#pol_position = interpolate.splev(xnovo, coef, der=0)
	
	#plt.plot(xnovo, pol_position, color='r', label='Posição por Cubic-Splines')
	
	plt.show()
	
	plt.title('Trajetória do objeto')
	plt.xlabel('Tempo (s)')
	plt.ylabel('Posição Y (px)')
	plt.scatter(Time, Y)
	plt.show()
	
	plt.title('Trajetória do objeto')
	plt.xlabel('Tempo (s)')
	plt.ylabel('Posição X (px)')
	plt.scatter(Time, X)
	plt.show()

def PendCollision(X, Y, Time, Objects): # Colisão pendular

	g = 9.78

	Energy = dict()
	Time = Time/1000
	plt.rcParams['figure.figsize'] = [11, 7]
	plt.title('Trajetória do objeto')
	plt.xlabel('Posição X (px)')
	plt.ylabel('Posição Y (px)')

	for i in range(Objects):
		Y['Object'+str(i)] = Y['Object'+str(i)]*-1
		Height = max(Y['Object'+str(i)]) - min(Y['Object'+str(i)])
		Energy['Object'+str(i)] = g*Height
		plt.scatter(X['Object'+str(i)], Y['Object'+str(i)], label=f'Objeto {i+1}')

	print(Energy)
	plt.grid()
	plt.legend()
	plt.show()