# tnreason for temporal clue game

Temporal clue games are constraint satisfaction problems where an assignment to variables in a murder case have to be found.
Constraints are given in the form of clues, stating specific facts that are known about the murder case.


## Toy example

Toy Example:

**Mystery Elements:**
- Suspects (WHO): \{Scarlet, Plum\}
- Locations (WHERE): \{Library, Kitchen\}
- Times (WHEN): \{Morning, Evening\}
- Methods (HOW): \{Poison, Knife\}


**Secret Solution:** (Plum, Kitchen, Evening, Knife)

**Clues:**
- Clue 1: "The murder happened in the Evening or Morning."
- Clue 2: "Miss Scarlet was in the Library during the Morning."
- Clue 3: "The murder was NOT in the Library."

[Solution notebook](https://colab.research.google.com/drive/1PQ-8J1ZptVWmI5_UMZJBWJfrE97KGvdN#scrollTo=Y6_D5f1TP8Hl)


## Further instances
[bradhilton/temporal-clue](https://github.com/bradhilton/temporal-clue/tree/main/puzzles)