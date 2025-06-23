python3 fes_2d.py COLVAR theta z -3.1415,3.1415 2.0,9.1
python3 analysis.py COLVAR z theta metad.rct 10 
python3 FE_time.py COLVAR z 2,8
