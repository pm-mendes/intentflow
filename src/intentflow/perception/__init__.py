"""
intentflow/perception — Phase 4 (points 1.1/2.1 du retour reviewer).

Chaîne multimédia réellement exécutée : screenshot/document scanné (image)
-> OCR (image_synthesis.py génère les fixtures, ocr.py extrait le texte) ->
le texte extrait devient un `RAGDocument` comme n'importe quel autre document
retrieved -> planner -> tool call -> décision du médiateur -> effet.

Avant cette phase, aucun de ces modules n'existait : `data/documents/` était
vide, aucune dépendance vision dans `pyproject.toml`. Le papier soumis
mentionnait OCR/screenshots dans l'introduction mais isolait explicitement
le canal textuel dans les expériences — c'est précisément le point 2.1 du
retour reviewer ("le fit multimédia est actuellement trop faible").
"""
