
# https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part3/nonstandard/#nuisance-parameter-impacts


combineTool.py -M Impacts -d ws.root -m 125 --doInitialFit --rMin -1 --rMax 2 --cminDefaultMinimizerStrategy 0 --robustFit 1 -t -1 --expectSignal=1
combineTool.py -M Impacts -d ws.root -m 125 --doFits --rMin -3 --cminDefaultMinimizerStrategy 0 --robustFit 1 -t -1  --expectSignal=1 --parallel 50
combineTool.py -M Impacts -d ws.root -m 125 -o ws.json
plotImpacts.py -i ws.json -o ws

