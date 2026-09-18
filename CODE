import random, json, csv, os, sys, statistics
from typing import List, Tuple, Dict, Any

def generate_data(n=400, seed=0):
    random.seed(seed)
    X, y = [], []
    for _ in range(n):
        study = max(0.0, min(12.0, random.gauss(5.0, 2.0)))
        attendance = max(40.0, min(100.0, random.gauss(80.0, 10.0)))
        sleep = max(4.0, min(10.0, random.gauss(7.0, 1.2)))
        prev = max(0.0, min(100.0, random.gauss(60.0, 15.0)))
        score = 0.5*study*8 + 0.2*attendance + 0.15*prev + sleep*1.5 + random.gauss(0,6)
        score = max(0.0, min(100.0, score))
        X.append([study, attendance, sleep, prev])
        y.append(score)
    return X, y

class Scaler:
    def __init__(self):
        self.means = []
        self.stds = []
    def fit(self, X: List[List[float]]):
        cols = list(zip(*X))
        self.means = [statistics.mean(c) for c in cols]
        self.stds = [statistics.pstdev(c) if statistics.pstdev(c)!=0 else 1.0 for c in cols]
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        out = []
        for row in X:
            out.append([(row[i]-self.means[i])/self.stds[i] for i in range(len(row))])
        return out
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    def save(self):
        return {"means": self.means, "stds": self.stds}
    def load(self, d):
        self.means = d.get("means", [])
        self.stds = d.get("stds", [])

class RulePredictor:
    def __init__(self, w=[8.0,0.3,2.0,0.4], bias=0.0):
        self.w = w
        self.b = bias
    def predict(self, X: List[List[float]]) -> List[float]:
        out = []
        for r in X:
            s = sum(r[i]*self.w[i] for i in range(4)) + self.b
            out.append(max(0.0, min(100.0, s)))
        return out
    def save(self):
        return {"type":"rule","w":self.w,"b":self.b}
    @staticmethod
    def load(d):
        return RulePredictor(w=d.get("w",[8.0,0.3,2.0,0.4]), bias=d.get("b",0.0))

class LinearGD:
    def __init__(self, n=4):
        self.w = [0.0]*n
        self.b = 0.0
    def predict(self, X):
        return [self.b + sum(self.w[i]*row[i] for i in range(len(row))) for row in X]
    def fit(self, X, y, lr=0.05, epochs=800, verbose=False):
        n = len(y)
        for ep in range(epochs):
            preds = self.predict(X)
            grad_w = [0.0]*len(self.w)
            grad_b = 0.0
            for i in range(n):
                e = preds[i]-y[i]
                for j in range(len(self.w)):
                    grad_w[j] += (2.0/n)*e*X[i][j]
                grad_b += (2.0/n)*e
            for j in range(len(self.w)):
                self.w[j] -= lr*grad_w[j]
            self.b -= lr*grad_b
            if verbose and ep%(epochs//4+1)==0:
                print("Epoch",ep)
    def save(self):
        return {"type":"linear_gd","w":self.w,"b":self.b}
    @staticmethod
    def load(d):
        m = LinearGD(n=len(d.get("w",[])))
        m.w = d.get("w",[])
        m.b = d.get("b",0.0)
        return m

def mse(a,b):
    return sum((a[i]-b[i])**2 for i in range(len(a)))/len(a) if a else 0.0
def mae(a,b):
    return sum(abs(a[i]-b[i]) for i in range(len(a)))/len(a) if a else 0.0

def save_json(path, obj):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2)
def load_json(path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def cli():
    print("Student Performance Predictor (pure Python)")
    model = None; scaler = None
    while True:
        print("\\nOptions:")
        print(" 1) Generate data & train")
        print(" 2) Load model (JSON)")
        print(" 3) Predict single (manual input)")
        print(" 4) Predict from CSV -> save predictions")
        print(" 5) Save current model")
        print(" 6) Exit")
        c = input("Choose: ").strip()
        if c=='1':
            X,y = generate_data(500, seed=1)
            # split
            cut = int(len(X)*0.82)
            X_train, X_test = X[:cut], X[cut:]
            y_train, y_test = y[:cut], y[cut:]
            print("Train size:",len(X_train),"Test size:",len(X_test))
            typ = input("Train which (r)ule or (l)inear? ").strip().lower()
            if typ=='r':
                model = RulePredictor()
                preds = model.predict(X_test)
                print("MSE:",mse(y_test,preds),"MAE:",mae(y_test,preds))
            else:
                scaler = Scaler()
                Xs = scaler.fit_transform(X_train)
                Xs_test = scaler.transform(X_test)
                lin = LinearGD(n=4)
                lin.fit(Xs, y_train, lr=0.05, epochs=900)
                model = lin
                preds = model.predict(Xs_test)
                print("MSE:",mse(y_test,preds),"MAE:",mae(y_test,preds))
        elif c=='2':
            p = input("Model file path: ").strip() or "model.json"
            if not os.path.exists(p):
                print("File not found")
                continue
            meta = load_json(p)
            if meta.get("model",{}).get("type")=="rule":
                model = RulePredictor.load(meta["model"])
            else:
                model = LinearGD.load(meta["model"])
            scaler = Scaler()
            if "scaler" in meta:
                scaler.load(meta["scaler"])
            print("Model loaded.")
        elif c=='3':
            try:
                s = float(input("Study hours: "))
                a = float(input("Attendance %: "))
                sl = float(input("Sleep hours: "))
                pm = float(input("Previous marks: "))
            except:
                print("Invalid input")
                continue
            row = [[s,a,sl,pm]]
            proc = row
            if scaler and scaler.means:
                proc = scaler.transform(row)
            if model is None:
                print("No model loaded. Using rule predictor.")
                model = RulePredictor()
            pred = model.predict(proc)[0]
            print("Predicted final marks:", round(pred,2))
        elif c=='4':
            csv_in = input("CSV input path: ").strip()
            csv_out = input("CSV output path: ").strip() or "predictions.csv"
            if not os.path.exists(csv_in):
                print("CSV not found")
                continue
            X_in = []
            with open(csv_in,"r",encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts)<4: continue
                    try:
                        row = [float(parts[0]),float(parts[1]),float(parts[2]),float(parts[3])]
                    except:
                        continue
                    X_in.append(row)
            proc = X_in
            if scaler and scaler.means:
                proc = scaler.transform(X_in)
            if model is None:
                model = RulePredictor()
            preds = model.predict(proc)
            with open(csv_out,"w",encoding="utf-8") as f:
                f.write("study,attendance,sleep,previous,predicted\\n")
                for r,p in zip(X_in,preds):
                    f.write(",".join(str(x) for x in r)+","+str(round(p,2))+"\\n")
            print("Wrote predictions to", csv_out)
        elif c=='5':
            out = {"model": model.save() if model else {}, "scaler": scaler.save() if scaler else {}}
            p = input("Save path (model.json): ").strip() or "model.json"
            save_json(p,out)
            print("Saved model to",p)
        elif c=='6':
            print("Goodbye")
            break
        else:
            print("Unknown option")

if __name__=="__main__":
    cli()
