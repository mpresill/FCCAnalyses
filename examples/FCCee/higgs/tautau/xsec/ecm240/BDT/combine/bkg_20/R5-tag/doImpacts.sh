
# https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part3/nonstandard/#nuisance-parameter-impacts


combineTool.py -M Impacts -d ws.root -m 125 --doInitialFit --rMin -10 --cminDefaultMinimizerStrategy 0 --robustFit 1 -t -1
combineTool.py -M Impacts -d ws.root -m 125 --doFits --rMin -10 --cminDefaultMinimizerStrategy 0 --robustFit 1 -t -1 --parallel 30
combineTool.py -M Impacts -d ws.root -m 125 -o ws.json
plotImpacts.py -i ws.json -o ws

