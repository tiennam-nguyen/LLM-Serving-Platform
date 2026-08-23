"""Workers module boundary.

Future responsibility:
    Defines runtime adapter boundaries for backend LLM serving engines
    (vLLM, mock workers, and future runtimes).

    Encapsulates engine-specific communication protocols so that upstream routing
    and gateway logic remain engine-agnostic.
"""
