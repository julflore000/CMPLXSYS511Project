from numpy.core.fromnumeric import size
from main import main
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from numpy import median
import matplotlib.patches as mpatches

#different ways to set up the company distribution
#Options: "Random", "Michigan" , "Washington"
initialDistribution = "Random"


#number of companies to start off with
numberCompanies = 100

#cost of forming a link with another company
urbanCost = 13#1.3

#by Product StdDeviation for the specific companies product distribution (equal across all companies)
byProductStdDev = .04

#contractor threshold (minimum probability overlap needs to be for the two companies to send/consume waste)
contractorThreshold = .06

#characteristic geographical range (higher the value the greater "clustering")
dNaught = 40

#whether to display scatterplot of network forming or run multiple simulations
display = False

#central agents (0->1): only for GIS where if greater than 0, a central waste management service will be set up in each county which accepts x % of waste from anyone
centralWasteAcceptance = 0

#d0 = 40, T0 = 0.06, c = 1.3, σ = 0.04, α = 2.4

#run main program once
if display:
    #just run simulation once
    main(initialDistribution,numberCompanies,urbanCost,byProductStdDev,
         contractorThreshold,dNaught,centralWasteAcceptance,display)
else:
    
    ##to do
    #1. Create plot points for varying cost parameter
    #2. Create plot points for varying dNaugh parameter
    #3. * Can also create in/out degrees for each scenario analysis (can include at extra portion of slides though for spcae)
 
    #for varying cost/dNaugh parameter plot point
    #set up plot display for centralWasteAcceptance variable changing
    xAxisTitle = "Dnaught Parameter"
    yAxisTitle = "Final Waste"
    plotTitle = f"{yAxisTitle} vs {xAxisTitle}"
    distributionPlots = ["Random","Michigan","Washington"]
    distributionColors = ["red","blue","green"]
    lines = []
    #get multiple runs data
    #setting up data plot array
    for distributionIndex in range(0,3):
        costParameterArray = np.arange(0,300,50)
        xScatterPlot = []
        finalWasteValues = []
        finalWasteStdDev = []
        finalPlotValues = []        
        for costParameter in costParameterArray:
            print(costParameter)
            for i in range(0,10):
                #plotDataset.append(main(initialDistribution,numberCompanies,urbanCost,byProductStdDev,contractorThreshold,dNaught,display))
                finalAnalysis = np.min(main(distributionPlots[distributionIndex],numberCompanies,
                                                    urbanCost,byProductStdDev,contractorThreshold,
                                                    costParameter,centralWasteAcceptance,display))
                xScatterPlot.append(round(costParameter,1))
                finalPlotValues.append(finalAnalysis)

        
        #plot data
        sns.pointplot(x=xScatterPlot, y=finalPlotValues,estimator=median,color=distributionColors[distributionIndex])#,label=distributionPlots[distributionIndex])

    #manually creating legend because of weird pointplot impact on handles
    legendList = []
    for distribution,color in zip(distributionPlots,distributionColors):
        legendList.append(mpatches.Patch(color=color, label=distribution))
        
    plt.legend(handles=legendList)

    plt.xlabel(xAxisTitle)
    plt.ylabel(yAxisTitle)
    plt.title(plotTitle)
    plt.show()
    
    
    #for box plot data
    yBoxPlotData = [[],[],[]]
    distributionPlots = ["random","Michigan","Washington"]
    
    for distributionIndex in range(0,3):
        for i in range(0,20):        
            yBoxPlotData[distributionIndex].append(
                np.min(main(initialDistribution,numberCompanies,
                                                  urbanCost,byProductStdDev,contractorThreshold,
                                                  dNaught,centralWasteAcceptance,display))
            )
    
    #plot data
    fig, ax = plt.subplots()
    ax.set_title('Final Waste per Scenario')
    ax.boxplot(yBoxPlotData)
    ax.set_xticklabels(distributionPlots)
    ax.set_ylabel("Final Waste")
    plt.show()
    
    
    #for centralWasteAcceptance plots
    #set up plot display for centralWasteAcceptance variable changing
    xAxisTitle = "Demand Decrease"
    yAxisTitle = "Final Waste"
    plotTitle = f"{yAxisTitle} vs {xAxisTitle} (Distribution: {initialDistribution})"
    #get multiple runs data
    #setting up data plot array
    centralWasteArray = np.arange(0,.9,.1)
    xScatterPlot = []
    finalWasteValues = []
    finalWasteStdDev = []
    
    for centralWasteAcceptance in centralWasteArray:
        finalWasteAnalysis = []
        print(centralWasteAcceptance)
        for i in range(0,10):
            #plotDataset.append(main(initialDistribution,numberCompanies,urbanCost,byProductStdDev,contractorThreshold,dNaught,display))
            finalWasteAnalysis = np.min(main(initialDistribution,numberCompanies,
                                                  urbanCost,byProductStdDev,contractorThreshold,
                                                  dNaught,centralWasteAcceptance,display))
            xScatterPlot.append(round(centralWasteAcceptance,1))
            finalWasteValues.append(finalWasteAnalysis)

    
    #plot data
    ax = sns.pointplot(x=xScatterPlot, y=finalWasteValues,estimator=median)
    plt.xlabel(xAxisTitle)
    plt.ylabel(yAxisTitle)
    plt.title(plotTitle)
    plt.show()
    

print("done")