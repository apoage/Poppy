"""Candle-language tokenizer panel (branch C, TASK_789/791/792/795).

A non-predictive, vol-unit, single-asset tokenizer that turns minute OHLC into a vocabulary of
recurring relative-shape "words". One shared substrate (volunit) + one shared base alphabet feed a
PANEL of isolated languages (L1 BPE / L2 ABBA / L3 SAX-grammar / L4 FSQ-VQ / L5 Matrix-Profile),
each judged by the same non-predictive battery (A0/A1/A2/A8/A9). See docs/candle_language/.

config.py is the interface contract; import shapes/constants from there, do not fork them.
"""
