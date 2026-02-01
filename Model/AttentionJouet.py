import numpy as np


def softmax(ligne):
    m = len(ligne)
    sommeExp = np.sum(np.exp(ligne))
    
    for j in range(0,m):
        ligne[j] = np.exp(ligne[j]) / sommeExp
    return ligne
     



def jouet_attention(n,d) : 
    X = np.random.randn(n, d)
    S = X@X.T
    A=np.zeros((n,n))
    for i in range(n):
        A[i]=softmax(S[i])
    O=A@X

    # --- AFFICHAGE DES VÉRIFICATIONS ---
    print("=== VÉRIFICATION DES TAILLES ===")
    print(f"Taille de X  : {X.shape}")
    print(f"Taille de S : {S.shape}")
    print(f"Taille de A   : {A.shape}")
    print(f"Taille de O   : {O.shape}")
    
    print("\n=== CONTENU DE LA MATRICE A (Ligne 3) ===")
    # On affiche la ligne 3 pour voir les poids d'attention
    print(A[3])
    print(f"Somme de la ligne 3 : {np.sum(A[3])}")
    
    print("\n=== COMPARAISON X vs O (Ligne 0) ===")
    print(f"Vecteur original X[0] :\n{X[0]}")
    print(f"Vecteur mis à jour O[0] :\n{O[0]}")
    
    return X, S, A, O
X, S, A, O = jouet_attention(10, 8)