"""Score by distance (``eval.by_distance``, opt-in; 18 Nov 2026; ``docs/distance_eval.md``).

A run's dev numbers are one mean per task. What separates the models of this program is *where*
an answer comes from: a fact 40 bytes back in the same window, 300 bytes back in an earlier one,
or 3,000 bytes back; a name the story used a paragraph ago. This package turns one evaluation
pass into those tables:

* :mod:`~hsn.eval.distance.capture`: the loss and the prediction of every loss position of a
  pass, by document and position (an observer of :func:`hsn.eval.evaluate.evaluate_windows`);
* :mod:`~hsn.eval.distance.recall`: the synthetic tasks: accuracy by recall distance (the
  shards' ``deps``), windows back, first against later query, value against ``NONE`` answers,
  the document's own values, and the fact's position against the distance;
* :mod:`~hsn.eval.distance.text`: text: bits per letter of repeated names and of content words
  by the distance of the previous mention;
* :mod:`~hsn.eval.distance.score`: the passes of a run (:class:`RunScorer`) and the trainer's
  observer (:class:`ByDistance`).

In a run, ``eval.by_distance.enabled=true`` writes the block ``by_distance`` into the run
record when training ends. For a run that exists, ``python -m hsn.eval.distance --run-dir RUN``
scores a checkpoint on the CPU or the card, with read-time overrides
(``--override model.reader_carry=256``) and further sets (``--set far=MANIFEST``).

Evaluation only. Nothing here changes training, a model or an existing metric: the pass is
:meth:`hsn.train.loop.Trainer.evaluate` itself, and the tables are sums over what it scored.
"""

from hsn.eval.distance.capture import Captured, PositionCapture
from hsn.eval.distance.recall import DISTANCE_KEYS, fact_positions, recall_table
from hsn.eval.distance.score import KEY, SCHEMA, ByDistance, RunScorer, SetScorer, render
from hsn.eval.distance.text import TextIndex, find_frequent_words, name_list, text_table

__all__ = [
    "DISTANCE_KEYS",
    "KEY",
    "SCHEMA",
    "ByDistance",
    "Captured",
    "PositionCapture",
    "RunScorer",
    "SetScorer",
    "TextIndex",
    "fact_positions",
    "find_frequent_words",
    "name_list",
    "recall_table",
    "render",
    "text_table",
]
