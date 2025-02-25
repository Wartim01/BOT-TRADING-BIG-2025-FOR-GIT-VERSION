import os

def is_text_file(filename, allowed_extensions=('.py', '.txt', '.json', '.md')):
    """
    Vérifie si le fichier est considéré comme textuel en fonction de son extension.
    On ignore les fichiers CSV.
    """
    filename = filename.lower()
    # On exclut les fichiers CSV
    if filename.endswith('.csv'):
        return False
    # On considère comme textuel uniquement les fichiers avec une extension autorisée
    return filename.endswith(allowed_extensions)

def lire_contenu_fichier(chemin_fichier, encodages=None):
    """
    Essaie d'ouvrir et de lire le fichier en utilisant plusieurs encodages.
    Renvoie le contenu si la lecture réussit, sinon renvoie une chaîne indiquant l'erreur.
    """
    if encodages is None:
        encodages = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'utf-16']
    for enc in encodages:
        try:
            with open(chemin_fichier, 'r', encoding=enc) as fichier:
                return fichier.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return f"Erreur lors de la lecture du fichier avec l'encodage {enc} : {e}"
    return "Erreur : Aucun des encodages n'a permis la lecture du fichier."

def regrouper_fichiers(dossier_source, fichier_sortie):
    """
    Parcourt récursivement le dossier_source et regroupe le contenu de chaque fichier
    dans fichier_sortie, uniquement pour les fichiers textuels spécifiés et qui ne sont pas des CSV.
    """
    with open(fichier_sortie, 'w', encoding='utf-8') as sortie:
        for chemin_dossier, _, fichiers in os.walk(dossier_source):
            for nom_fichier in fichiers:
                if not is_text_file(nom_fichier):
                    continue  # Ignore les fichiers non textuels ou les CSV
                chemin_fichier = os.path.join(chemin_dossier, nom_fichier)
                sortie.write("Chemin complet : {}\n".format(chemin_fichier))
                sortie.write("Nom du fichier : {}\n".format(nom_fichier))
                sortie.write("Contenu :\n")
                contenu = lire_contenu_fichier(chemin_fichier)
                sortie.write(contenu)
                sortie.write("\n" + "-" * 80 + "\n\n")

if __name__ == "__main__":
    dossier_source = "C:\\Users\\timot\\OneDrive\\Bureau\\BOT TRADING BIG 2025\\trading_bot"
    fichier_sortie = input("Entrez le nom du fichier de sortie (ex: Botdetrading.txt) : ")
    regrouper_fichiers(dossier_source, fichier_sortie)
    print("Le contenu de tous les fichiers textuels (hors CSV) a été regroupé dans '{}'.".format(fichier_sortie))
