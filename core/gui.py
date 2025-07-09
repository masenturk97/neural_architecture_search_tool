import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from core.nn_search_engine import NNSearchEngine
import threading
from tkinter import filedialog as fd

def run():
    def combobox(root, text_variable, values, row, column):
        combobox = ttk.Combobox(root, textvariable=text_variable, values=values, state="readonly", width=10)
        combobox.set(values[0])
        combobox.grid(row=row, column=column, padx=5, pady=0)

    def startSearchThreading():
        if(datasetSelection_value.get() == ""):
            messagebox.showwarning(title="Warning", message="You must select a CSV file!")
        else:
            threading.Thread(target=startSearch).start()

    def startSearch():
        param = {
            "dataset": datasetSelection_value.get(),
            "populationSize": populationSize_value.get(),
            "maxGenerationCount": maxGenerationCount_value.get(),
            "taskType": taskType.get()
        }
        answer = messagebox.askokcancel(message="Are you sure to start development process?")

        if answer:
            startSearchButton.config(state="disabled")
            saveButton.config(state="disabled")
            tree.delete(*tree.get_children())
            bestResultTree.delete(*bestResultTree.get_children())
            progressbar.start()
            
            search_engine = NNSearchEngine(param)
            
            global lastBestModel
            lastBestModel = None
            lastGenerationCount = 0
            for iteration_models in search_engine:
                for model in iteration_models:
                    model = model.toDict()
                    values_to_insert = [search_engine.generationCount, model["fitnessScore"], model["architecture"], model["history"]]
                    tree.insert("", 'end', values=values_to_insert)
                bestModel = search_engine.finalBestFoundModel()
                if (lastBestModel is None):
                    lastBestModel = bestModel
                    lastGenerationCount = search_engine.generationCount
                else:
                    if(param["taskType"]==1):
                        if(bestModel.fitnessScore < lastBestModel.fitnessScore):
                            lastBestModel = bestModel
                            lastGenerationCount = search_engine.generationCount
                    else:
                        if(bestModel.fitnessScore > lastBestModel.fitnessScore):
                            lastBestModel = bestModel
                            lastGenerationCount = search_engine.generationCount
                    
            bestModel_dict = lastBestModel.toDict()
            values_to_insert = [lastGenerationCount, bestModel_dict["fitnessScore"], bestModel_dict["architecture"], bestModel_dict["history"]]
            bestResultTree.insert("", 'end', values=values_to_insert)
            progressbar.stop()
            messagebox.showinfo(title="Info", message="Model development process has completed.")
            startSearchButton.config(state="active")
            saveButton.config(state="active")

    window = tk.Tk()
    window.resizable(False, False)
    window.geometry("1200x700")
    window.title("Automatic Artificial Neural Network Generation Tool")

    paramsFrame = tk.Frame(window)
    paramsFrame.pack(fill='x', padx=10, pady=10)

    taskTypeFrame = tk.LabelFrame(paramsFrame, text="Task Type", padx=20, pady=10)
    taskTypeFrame.pack(side="left", fill="y")
    taskType = tk.IntVar()
    taskType.set(1)
    ttk.Radiobutton(taskTypeFrame, text="Regression", width=21, variable=taskType, value=1).pack()
    ttk.Radiobutton(taskTypeFrame, text="Binary Classification", width=21, variable=taskType, value=2).pack()
    ttk.Radiobutton(taskTypeFrame, text="Multi Classification", width=21, variable=taskType, value=3).pack()

    datasetSelectionFrame = tk.LabelFrame(paramsFrame, text="Dataset File Selection", padx=20, pady=15)
    datasetSelectionFrame.pack(side="left", fill="y", padx=5)
    datasetSelection_value = tk.StringVar()
    selectedFileName = tk.StringVar()
    selectedFileName.set("File is not selected.")
    def select_file():
        filename = fd.askopenfilename(
            title="Select CSV File",
            filetypes=(
                ("CSV files", "*.csv"),
            )
        )
        datasetSelection_value.set(filename)
        selectedFileName.set(filename.split("/")[-1])
    datasetSelectButton = ttk.Button(datasetSelectionFrame, text="Select CSV File", command=select_file)
    datasetSelectButton.pack()
    ttk.Label(datasetSelectionFrame, textvariable=selectedFileName, anchor="center", width=20).pack()

    geneticParamsFrame = tk.LabelFrame(paramsFrame, text="Genetic Algorithm Parameters", padx=20, pady=15)
    geneticParamsFrame.pack(side="left", fill="y")
    populationSize_value = tk.StringVar()
    ttk.Label(geneticParamsFrame, text="Count of Individuals in Population:", width=32).grid(row=0, column=0)
    combobox(geneticParamsFrame, populationSize_value, [*range(4,11,2)], row=0, column=1)
    maxGenerationCount_value = tk.StringVar()
    ttk.Label(geneticParamsFrame, text="Maximum Count of Generations:", width=32).grid(row=1, column=0)
    combobox(geneticParamsFrame, maxGenerationCount_value, [*range(1,11)], row=1, column=1)


    buttonsFrame = tk.LabelFrame(paramsFrame, text="Operations", padx=10, pady=10)
    buttonsFrame.pack(side="left", fill="y", padx=5)
    startSearchButton = ttk.Button(buttonsFrame, text="START", command=startSearchThreading)
    startSearchButton.grid(row=0, column=0, pady=5)
    progressbar = ttk.Progressbar(buttonsFrame, orient=tk.HORIZONTAL, mode="indeterminate")
    progressbar.grid(row=1, column=0)

    searchResultsframe = tk.LabelFrame(window, text="Development Results", padx=15, pady=15)
    searchResultsframe.pack(fill=tk.BOTH, expand=True, padx=10)
    tree = ttk.Treeview(searchResultsframe, columns=("1", "2", "3", "4"), show="headings")
    tree.column("1", width = 60, anchor ='c')
    tree.heading("1", text ="Generation")
    tree.column("2", width = 100, anchor ='c')
    tree.heading("2", text ="Fitness Score")
    tree.column("3", width = 350, anchor ='c')
    tree.heading("3", text ="Model Architecture")
    tree.column("4", width = 220, anchor ='c')
    tree.heading("4", text ="Model Metric Results")
    tree_scroll_y = ttk.Scrollbar(searchResultsframe, orient="vertical", command=tree.yview)
    tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=tree_scroll_y.set)
    tree.pack(fill=tk.BOTH, expand=True)

    bestResultframe = tk.LabelFrame(window, text="Best Development Result", padx=20, pady=10)
    bestResultframe.pack(fill='x', padx=10)
    bestResultTree = ttk.Treeview(bestResultframe, columns=("1", "2", "3", "4"), show="headings", height=1)
    bestResultTree.column("1", width = 60, anchor ='c')
    bestResultTree.heading("1", text ="Generation")
    bestResultTree.column("2", width = 100, anchor ='c')
    bestResultTree.heading("2", text ="Fitness Score")
    bestResultTree.column("3", width = 350, anchor ='c')
    bestResultTree.heading("3", text ="Model Architecture")
    bestResultTree.column("4", width = 220, anchor ='c')
    bestResultTree.heading("4", text ="Model Metric Results")
    bestResultTree.pack(fill=tk.BOTH, expand=True)

    def saveModel():
        saveButton.config(state="disabled")
        folderName = fd.askdirectory(
            title="Select Folder"
        )
        lastBestModel.model.save(f"{folderName}/bestModel.{fileType.get()}")
        messagebox.showinfo(title="Info", message="Model has saved.")
        saveButton.config(state="active")

    fileTypeFrame = tk.LabelFrame(bestResultframe, text="Model Saving Settings", padx=20, pady=10)
    fileTypeFrame.pack()
    fileType = tk.StringVar()
    fileType.set("keras")
    ttk.Radiobutton(fileTypeFrame, text=".h5", width=21, variable=fileType, value="h5").pack(side="left")
    ttk.Radiobutton(fileTypeFrame, text=".keras", width=21, variable=fileType, value="keras").pack(side="left")
    saveButton = ttk.Button(fileTypeFrame, text="Save Model", command=saveModel)
    saveButton.pack(side="left")
    saveButton.config(state="disabled")

    ttk.Label(window, text="").pack(anchor='w', padx=1, pady=1)

    window.mainloop()