import os, sys
sys.path.insert(0, os.environ.get('EXCALIDRAW_GEN_DIR', '.'))  # directory containing excalidraw_gen.py
from excalidraw_gen import Diagram
d=Diagram()
d.text("Eval noise setup: one model, many 200-question tests",0,-70,size=26)
a=d.box("AG News test split\n(7,600 articles)",0,0,color="gray")
b=d.box("Qwen3-0.6B / 1.7B\nchat prompt, prefill\n'The answer is' -> A/B/C/D",330,0,color="blue")
c=d.box("1 deterministic\nprediction per article",680,0,color="green")
d.arrow(a,b,label="article"); d.arrow(b,c,label="argmax logit")
e=d.box("Corrected rerun: 20 overlapping\nsubsets of a 1,000 pool",260,230,color="yellow")
f=d.box("Fresh-sample arm: 20 DISJOINT\nsets from 4,000 fresh items",760,230,color="orange")
g=d.box("190 pairs:\nhow often does |diff| >= 2 pp?",500,430,color="violet")
d.arrow(c,e); d.arrow(c,f); d.arrow(e,g); d.arrow(f,g)
d.export_png(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figures","diagram_setup.png"),scale=2)
