import matplotlib.pyplot as plt
import random
import numpy as np
from statistics import NormalDist
import math
import shapefile as shp  # Requires the pyshp package for GIS work
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import LinearLocator



def calculateUtility(overlap,firstPointIndex,secondPointIndex):
    """calculates the utility that the company pair has 
        According to formula: overlapProbability - linkCost*(pairDistance/systemMaxDistance)
    """ 
    return overlap - linkCost*(totalDistanceArray[firstPointIndex][secondPointIndex])/(systemMaxDistance)


def calculateGeographicInteractionPotential():
    """calculates the geographic potential for each company interaction pair (will be run after calculate distances )
    """  
    for firstPointIndex in range(0,len(totalDistanceArray[0])):
        for secondPointIndex in range(0,len(totalDistanceArray[0])):
            #calculate eculadian distance from two points
            geographicPotentialArray[firstPointIndex][secondPointIndex] = math.exp(-1*totalDistanceArray[firstPointIndex][secondPointIndex]/charGeoRange)




def calculateDistances():
    """calculates the 2d distance from each company and stores in the

    Returns: nothing but updates totalDistance array with correct company decisions
    """    
    #method calculates the 2d distance from each company and stores in the
    #2D array totalDistance (row represents the firstPoint and col represents the second col)
    
    
    for firstPointIndex in range(0,len(totalDistanceArray[0])):
        for secondPointIndex in range(0,len(totalDistanceArray[0])):
            #calculate eculadian distance from two points
            totalDistanceArray[firstPointIndex][secondPointIndex] = math.sqrt(math.pow(xLoc[firstPointIndex]-xLoc[secondPointIndex],2)+ math.pow(yLoc[firstPointIndex]-yLoc[secondPointIndex],2))

def plotCountyBounds(sf,axis,washingtonPlot):
    """plots the county boundary outlines for selected state (right now only have functionality for MI)
    sf: shapefile of region
    washingtonPlot (bool): whether to switch plotting method to washington
    """
    for shape in sf.shapeRecords():
        if washingtonPlot:
            if(shape.record.STATEFP != '53'):
                #unique identifier to plot only washington in case
                continue    
        if(len(shape.shape.parts)>= 2):
            priorPoint = 0
            for partPoint in shape.shape.parts:
                
                #plotting data
                x = [i[0] for i in shape.shape.points[priorPoint:partPoint]]
                y = [i[1] for i in shape.shape.points[priorPoint:partPoint]]
                axis.plot(x,y,color="black")
                priorPoint = partPoint
        else:
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            axis.plot(x,y,color="black")
    
def setup(distribution,numberCompanies):
    """initialize the program with each unique companies position and then calculate the respective distances of each company from one another

    Args:
        distribution (str): distribution of company locations
        numberCompanies (int): how many companies to create
    """ 
    
    
    if (distribution == "Random") or (distribution == "handOfGod"):
        #assign randomly the initial locations of each company
        for i in range(0,numberCompanies):
            xLoc.append(random.random()*10)
            yLoc.append(random.random()*10)
    elif distribution == "Michigan":
        #read in the shapefile for the GIS county outlines
        sf = shp.Reader("../data/michiganShapefile/Counties_(v17a).shp")
        
        #get out the GIS data for the michigan company locations (where counties are) and select the respective unique company locations
        countyData = pd.read_excel("../data/statePopExcels/michiganPop.xlsx")

        populationArray = np.array(countyData["Cumulative Pop"])
        latArray = np.array(countyData["Lat"])
        lonArray = np.array(countyData["Lon"])
        
        #creating countyDistributions
        countyDistributions = []
        for i in range(0,len(populationArray)):
            countyDistributions.append(random.uniform(0,1))
        
        #for assigning similar or different distributions
        countyID = []
        
        #randomly select a county based on the number of people in each county for each company
        for i in range(0,numberCompanies):
            randomValue = random.randint(0,max(populationArray))
            closestIndex = np.searchsorted(populationArray,[randomValue,],side='right')[0]
            countyID.append(closestIndex)
            xLoc.append(lonArray[closestIndex])
            yLoc.append(latArray[closestIndex])
    elif distribution == "Washington":
         #read in the shapefile for the GIS county outlines
        sf = shp.Reader("../data/entireCountryShapefile/tl_2019_us_county.shp")
        
        #get out the GIS data for the michigan company locations (where counties are) and select the respective unique company locations
        countyData = pd.read_excel("../data/statePopExcels/washingtonPop.xlsx")

        populationArray = np.array(countyData["Cumulative Pop"])
        latArray = np.array(countyData["Lat"])
        lonArray = np.array(countyData["Lon"])
        
        #randomly select a county based on the number of people in each county for each company
        for i in range(0,numberCompanies):
            randomValue = random.randint(0,max(populationArray))
            closestIndex = np.searchsorted(populationArray,[randomValue,],side='right')[0]
            xLoc.append(lonArray[closestIndex])
            yLoc.append(latArray[closestIndex])       
    #correctly fill in distances
    calculateDistances()
    
    #calculate geographic potential function
    calculateGeographicInteractionPotential()
    
    #create normal distributions for each company's input (demand) and output (waste)
    for companyIndex in range(0,numberCompanies):
        if((distribution == "Michigan") | (distribution == "Washington")):
            inputGaussianDistribution.append(NormalDist(mu=random.uniform(0,1), sigma=stdDev))
            outputGaussianDistribution.append(NormalDist(mu=random.uniform(0,1), sigma=stdDev))
        else:
            inputGaussianDistribution.append(NormalDist(mu=random.uniform(0,1), sigma=stdDev))
            outputGaussianDistribution.append(NormalDist(mu=random.uniform(0,1), sigma=stdDev))


    global systemMaxDistance
    #system max distance between two companies (used in utility function)-set after totalDistanceArray intialized
    systemMaxDistance = np.amax(totalDistanceArray)
    
    #display initial setup of companies
    if(displayVisuals):
        global fig,ax
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.scatter(xLoc,yLoc)
        if(distribution == "Michigan"):
            plotCountyBounds(sf,ax,False)
        elif(distribution == "Washington"):
            plotCountyBounds(sf,ax,True)
        fig.tight_layout()
        ax.axis("off")
        fig.show()

    
def step():
    """Takes one step forward in simulation
    """    
    
    #randomly selecting a company from the smallest minimum link companies
    minimumLinkCompanyIndices = [i for i in range(len(companyNumberLinks)) if companyNumberLinks[i] == min(companyNumberLinks)]

    #selecting randomly from the minimum link companies (referred to as the contractor) (i.e. home base)
    contractorIndex = minimumLinkCompanyIndices[random.randint(0,len(minimumLinkCompanyIndices)-1)]
    
    #conduct first run through in selecting companies based on geographic closeness using potential function
    #generate random values to see if under given value
    randomValueList = np.random.random_sample(size = len(companyNumberLinks))
    
    #selecting companies based on geographcPotentialFunction (faster method would be to not generate an extra random number)
    geographicInteractionIndices = [i for i in range(len(companyNumberLinks)) if((geographicPotentialArray[contractorIndex][i] >= randomValueList[i]) & (i != contractorIndex))]
    
    
    #selecting compatible companies based on overlapProducts value being greater than contractorThreshold and there is still free demand and waste left(second to final step)
    
    compatibleCompanyIndices = [i for i in geographicInteractionIndices if((outputGaussianDistribution[contractorIndex].overlap(inputGaussianDistribution[i]) >= contractorThreshold) and (companyInteractions[contractorIndex][i] == False) and(outputGaussianDistribution[contractorIndex].overlap(inputGaussianDistribution[i]) <= min(outputWasteArray[1][contractorIndex],inputDemandArray[1][i])))]
    
    #if not comptabile companies after geographic and product overlap, simply move on
    if len(compatibleCompanyIndices) == 0:
        #for plotting data want total waste amount
        totalWaste.append(np.sum(outputWasteArray[1]))
        
        #satisfying simulation run criteria
        outputWasteArray[0] = outputWasteArray[1]
        inputDemandArray[0] = inputDemandArray[1]
        return

    #final step: selecting company pair which maximizes utility!
    utilityList = [calculateUtility(outputGaussianDistribution[contractorIndex].overlap(inputGaussianDistribution[i]),contractorIndex,i) for i in compatibleCompanyIndices]
    
    #find max utility
    optimizedSecondCompanyIndex = compatibleCompanyIndices[utilityList.index(max(utilityList))]
    

    #make each company's number of links increase by 1
    companyNumberLinks[contractorIndex] += 1
    companyNumberLinks[optimizedSecondCompanyIndex] += 1
    
    #switch that each company pair has interacted
    companyInteractions[contractorIndex][optimizedSecondCompanyIndex] = 1
    
    #Important: we are not changing the input/output distributions after agreement occurs
    if(displayVisuals):
        ##add directed edge on graph from contractor to second company
        ax.arrow(xLoc[contractorIndex],yLoc[contractorIndex],xLoc[optimizedSecondCompanyIndex]-xLoc[contractorIndex],yLoc[optimizedSecondCompanyIndex]-yLoc[contractorIndex],
                color="black",head_width = 0.1,head_length = 0.15,length_includes_head=True)
        fig.canvas.draw()
        fig.canvas.flush_events()

    #subtract waste output from two interactin companies
    outputWasteArray[0][contractorIndex] = outputWasteArray[1][contractorIndex] 
    outputWasteArray[1][contractorIndex] -= outputGaussianDistribution[contractorIndex].overlap(inputGaussianDistribution[optimizedSecondCompanyIndex])

    inputDemandArray[0][optimizedSecondCompanyIndex] = inputDemandArray[1][optimizedSecondCompanyIndex]    
    inputDemandArray[1][optimizedSecondCompanyIndex] -= outputGaussianDistribution[contractorIndex].overlap(inputGaussianDistribution[optimizedSecondCompanyIndex])    
    #add link cost to running total
    global totalLinkCost
    totalLinkCost += linkCost*totalDistanceArray[contractorIndex][optimizedSecondCompanyIndex]   
    
    #add waste amount to totalWaste Array
    totalWaste.append(np.sum(outputWasteArray[1]))
    
    
def main(initialDistribution,numberCompanies,linkFormationCost,byProductStdDev,acceptanceThreshold,characteristicGeographicConstant,centralWasteAcceptance,display):
    ### DECLARATIONS ###
    
    #setting up unique 1d arrays
    global companyNumberLinks,xLoc,yLoc,totalWaste
    
    #number of links that each company has
    companyNumberLinks = np.zeros(numberCompanies)
    
    #x and y locations of each company
    xLoc = []
    yLoc = []
    
    #tracks the totalwaste progression over simulations time
    totalWaste = []
    
    #setting up unique 2d arrays
    global totalDistanceArray,geographicPotentialArray,inputGaussianDistribution,outputGaussianDistribution,outputWasteArray,inputDemandArray,companyInteractions
    
    #2D distance from each company
    totalDistanceArray = np.zeros((numberCompanies,numberCompanies))
    
    #2d geographic potential function which incorporates characteristic geographical input (dNaught) for probability of companies interacting
    geographicPotentialArray = np.zeros((numberCompanies,numberCompanies))
    
    #2d array of the input Gaussian distribution (demand)
    inputGaussianDistribution = []
    
    #2d array of the output Gaussian distribution (waste)
    outputGaussianDistribution = []
    
    #2d array containing the total output waste at time t (1st row), and t+1 (2nd row) for each company
    outputWasteArray = np.ones((2,numberCompanies))
    
    #2d array containing the total input demand at time t (1st row), and t+1 (2nd row) for each company
    inputDemandArray = np.ones((2,numberCompanies))
    
    #2d array that defines (1 true, 0 false) whether that company pair already has a link
    companyInteractions = np.zeros((numberCompanies,numberCompanies))
    
    
    #defining constants (see inputs.py for info on constants)
    global linkCost,stdDev,contractorThreshold,charGeoRange,totalLinkCost,displayVisuals
    linkCost = linkFormationCost
    stdDev = byProductStdDev
    contractorThreshold = acceptanceThreshold
    charGeoRange = characteristicGeographicConstant
    
    #running calculation of the total cost of creating all the links
    totalLinkCost = 0

    #whether to graphically display network formation
    displayVisuals = display
    ### END DECLARATIONS ###
   
    
    ### SETUP SIMULATION ###
    
    #setting up initial locations
    setup(initialDistribution,numberCompanies)
    
    ### END SETUP ###
    if(centralWasteAcceptance > 0):
        
        #for decreasing waste
        #outputWasteArray -= centralWasteAcceptance
        
        #for increasing demand analysis
        inputDemandArray += centralWasteAcceptance
    

    ### START SIMULATION RUN ###
    #initiate simulation
    for i in range(0,2):
        step()
        
    #stopping critera, taken from paper
    epsilon = .001
    while(np.sum(outputWasteArray[0]-outputWasteArray[1])/numberCompanies > epsilon):
        step()
     
        
   
    #print("SIMULATION RUN DONE")
    ### END SIMULATION RUN ###

    ### COLLECT DATA/RUN INFO for wanting display information
    
    #metrics to collect: total waste, total cost
    if(display):
        
        averageWasteLeft = np.average(outputWasteArray)
        
        print(f"Average waste left: {averageWasteLeft}")
        print(f"Average number of links: {np.average(companyNumberLinks)}")
        print(f"Total link development cost: {totalLinkCost}")
        
        
        ### total waste vs time graph ###
        #plt.plot(np.array(totalWaste)/numberCompanies)
        #plt.show()
        
        ### total degree frequency graph ###
        plt.subplot(1,1,1)
        plt.hist(companyNumberLinks,bins=np.arange(0, companyNumberLinks.max() + 1.5) - 0.5, density=True)
        plt.title(f"Degree Density Across Companies (Distribution: {initialDistribution})")
        plt.xlabel("Total Degree")
        plt.ylabel("Percentage (%)")
        plt.show()
        
        
        ### degree frequency graphs ###

        ##creating input output degree distribution arrays
        # companyInteractions gives you edge from (row) to (col) cell
        # so summing up by row gives you total out going edges or out degree
        # and then summing up by col gives you total in going edge or in degree
        
        plotInDegree = []
        plotOutDegree = []
        for i in range(0,numberCompanies):
            plotInDegree.append(int(np.sum(np.transpose(companyInteractions)[i])))
            plotOutDegree.append(int(np.sum(companyInteractions[i])))
        
        #converting to numpy    
        plotInDegree = np.array(plotInDegree)
        plotOutDegree = np.array(plotOutDegree)
                    
        plt.subplot(1,2,1)
        plt.hist(plotInDegree,bins=np.arange(0, np.array(plotInDegree).max() + 1.5) - 0.5, density=True)
        plt.title(f"In degree Across Companies (Distribution: {initialDistribution})")
        plt.ylabel("Percentage (%)")
        plt.xlabel("In Degree")
        
        plt.subplot(1,2,2)
        plt.hist(plotOutDegree,bins=np.arange(0, np.array(plotOutDegree).max() + 1.5) - 0.5, density=True)
        plt.title(f"Out degree Across Companies (Distribution: {initialDistribution})")
        plt.ylabel("Percentage (%)")
        plt.xlabel("Out Degree")
        plt.show()
        
        
        ### heat map plotting ###
        '''
        #first convert to numpy array
        plotInDegree = np.array(plotInDegree)
        plotOutDegree = np.array(plotOutDegree)
        
        #create empty heatmap array
        heatmapDf = np.zeros((np.max(plotOutDegree)+1,np.max(plotInDegree)+1))

        #fill in heatmap amount
        for xIndex,yIndex in zip(plotOutDegree,plotInDegree):
            heatmapDf[xIndex][yIndex] += 1

        #display heat map
        ax = sns.heatmap(heatmapDf, linewidth=0.5)
        ax.invert_yaxis()

        plt.title("Out degree vs In degree heatmap")
        plt.xlabel("In degree")
        plt.ylabel("Out degree")
        plt.show()
        
        print(f"Average difference between in degree vs out degree: {np.average(plotInDegree-plotOutDegree)}")
        print("done")
        '''

        ### 3D Visualization ### 
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        xLoc = np.array(xLoc)
        yLoc = np.array(yLoc)

        xRange = np.arange(min(xLoc)-2.5, max(xLoc)+2.5, 0.25)
        yRange = np.arange(min(yLoc)-2.5, max(yLoc)+2.5, 0.25)
        X, Y = np.meshgrid(xRange, yRange)

        zData = companyNumberLinks
        
        Z = (X == -1).astype(int)
        for zIndex in range(0,len(zData)):
            value3dIndex = [(np.abs(yRange - yLoc[zIndex])).argmin(),(np.abs(xRange - xLoc[zIndex])).argmin()]
            Z[value3dIndex[0]][value3dIndex[1]] = zData[zIndex]


        # Plot the surface.
        heatmap = ax.contourf(X, Y, Z, zdir='z')
        
        if(initialDistribution == "Michigan"):
            plotCountyBounds(shp.Reader("../data/michiganShapefile/Counties_(v17a).shp"),ax,False)
        elif(initialDistribution == "Washington"):
            plotCountyBounds(shp.Reader("../data/entireCountryShapefile/tl_2019_us_county.shp"),ax,True)
            
        '''cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.ocean)
        cset = ax.contourf(X, Y, Z, zdir='x', offset=-5, cmap=cm.ocean)
        cset = ax.contourf(X, Y, Z, zdir='y', offset=5, cmap=cm.ocean)'''
        fig.colorbar(heatmap,shrink=.75)
        ax.axis("off")
        plt.show()
        ### end data visualization ###
    else:
        return np.array(totalWaste)/numberCompanies
    
    
    ### END RUN ###
    