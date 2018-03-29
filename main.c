/*
    Author: Johnathan M Melo Neto (jmmn.mg@gmail.com)
    Related paper: "Hybridization of Cartesian Genetic Programming and Differential Evolution 
        for Generating Classifiers based on Neural Networks"

    This file is an adapted version of CGP-Library
    Copyright (c) Andrew James Turner 2014, 2015 (andrew.turner@york.ac.uk)
    The original CGP-Library is available in <http://www.cgplibrary.co.uk>    
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "cgpdelib.h"

double accuracy(struct parameters *, struct chromosome *, struct dataSet *);

int main(void)
{
    struct parameters *params = NULL;
    struct dataSet *mainData = NULL;

    // Insert the desired dataset here
    mainData = initialiseDataSetFromFile("./dataSets/iris.txt");

    // Initialize general parameters
    int numInputs = 4;  // attributes
    int numOutputs = 3; // classes
	
    // Percentage of the sample size utilized: 0 < percentage <= 1
    double percentage = 1.00;	

    int numThreads = 10;

    int numNodes = 500;
    int nodeArity =  20;
    double weightRange = 5; 
    double mutationRate = 0.05;

    double CR = 0.90;
    double F = 0.70;

    // Set general parameters
    params = initialiseParameters(numInputs, numNodes, numOutputs, nodeArity);
    setCustomFitnessFunction(params, accuracy, "Accuracy");
    addNodeFunction(params, "sig");
    setMutationType(params, "probabilistic");
    setConnectionWeightRange(params, weightRange);
    setMutationRate(params, mutationRate);
    setNumThreads(params, numThreads);
    setCR(params, CR);
    setF(params, F);

    // CGPANN specific parameters
    int numGens_CGP = 50000;
	
    // CGPDE-IN specific parameters
    int numGens_IN = 64;
    int NP_IN = 10;
    int maxIter_IN = 400;

    setNP_IN(params, NP_IN);
    setMaxIter_IN(params, maxIter_IN);

    // CGPDE-OUT (T and V) specific parameters
    int numGens_OUT = 40000;
    int NP_OUT = 20;
    int maxIter_OUT = 2570;

    setNP_OUT(params, NP_OUT);
    setMaxIter_OUT(params, maxIter_OUT);

    // Open text files to store the results
    FILE *f_CGP = fopen("./results/cgpann.txt", "w");
    FILE *f_IN = fopen("./results/cgpde_in.txt", "w");
    FILE *f_OUT_T = fopen("./results/cgpde_out_t.txt", "w");
    FILE *f_OUT_V = fopen("./results/cgpde_out_v.txt", "w");

    if (f_CGP == NULL || f_IN == NULL || f_OUT_T == NULL || f_OUT_V == NULL)
    {
        printf("Error opening files!\n");
        exit(1);
    }

    // Header of the text files (to track the result of each independent run)
    fprintf(f_CGP, "i,\tj,\taccuracy\n");
    fprintf(f_IN, "i,\tj,\taccuracy\n");
    fprintf(f_OUT_T, "i,\tj,\taccuracy\n");
    fprintf(f_OUT_V, "i,\tj,\taccuracy\n");

    // Initialize the experiments
    printf("TYPE\t\ti\tj\tFIT\n\n");
    int i, j;
    
    for(i = 0; i < 3; i++) // 3 independent cross-validations
    {
        // Set seed (for reproducibility purpose)
        unsigned int seed = i + 50;
        shuffleData(mainData, &seed);
        struct dataSet * reducedData = reduceSampleSize(mainData, percentage);        
	struct dataSet ** folds = generateFolds(reducedData);

        #pragma omp parallel for default(none), private(j), shared(i,params,folds,numGens_CGP,numGens_IN,numGens_OUT,f_CGP,f_IN,f_OUT_T,f_OUT_V,NP_OUT), schedule(dynamic), num_threads(numThreads)
        for(j = 0; j < 10; j++) // stratified 10-fold cross-validation
        {
            // Set different seed for each independent run (for reproducibility purpose)
            unsigned int seed = (i*10)+j+5;

            // Build training, validation, and testing sets
            int * training_index = (int*)malloc(7*sizeof(int));
            int * validation_index = (int*)malloc(2*sizeof(int));
            int testing_index = j;
            getIndex(training_index, validation_index, testing_index, &seed);

            struct dataSet *trainingData = getTrainingData(folds, training_index);         
            struct dataSet *validationData = getValidationData(folds, validation_index);
            struct dataSet *testingData = getTestingData(folds, testing_index);

            // Save training, validation, and testing sets
            #pragma omp critical
            {
                char filename[100];
                char buf_i[10];
                char buf_j[10];
                memset(filename, '\0', sizeof(char)*100);
                memset(buf_i, '\0', sizeof(char)*10);
                memset(buf_j, '\0', sizeof(char)*10);
                snprintf(buf_i, 10, "%d", i);
                snprintf(buf_j, 10, "%d", j);
                strcat(filename, "./results/TRN/TRN_");
                strcat(filename, buf_i);
                strcat(filename, "_");
                strcat(filename, buf_j);
                strcat(filename, ".txt");
                saveDataSet(trainingData, filename);

                memset(filename, '\0', sizeof(char)*100);
                strcat(filename, "./results/VLD/VLD_");
                strcat(filename, buf_i);
                strcat(filename, "_");
                strcat(filename, buf_j);
                strcat(filename, ".txt");
                saveDataSet(validationData, filename);

                memset(filename, '\0', sizeof(char)*100);
                strcat(filename, "./results/TST/TST_");
                strcat(filename, buf_i);
                strcat(filename, "_");
                strcat(filename, buf_j);
                strcat(filename, ".txt");
                saveDataSet(testingData, filename);
            }

            // Run CGPANN 
            struct chromosome * bestChromo = runCGP(params, trainingData, validationData, numGens_CGP, &seed);
            setChromosomeFitness(params, bestChromo, testingData);
            double testFitness_CGP = getChromosomeFitness(bestChromo);  
            printf("CGPANN\t\t%d\t%d\t%.4lf\n", i, j, -testFitness_CGP);
            freeChromosome(bestChromo);  

            // Run CGPDE-IN
            bestChromo = runCGPDE_IN(params, trainingData, validationData, numGens_IN, &seed); 
            setChromosomeFitness(params, bestChromo, testingData);
            double testFitness_IN = getChromosomeFitness(bestChromo);  
            printf("CGPDE-IN\t%d\t%d\t%.4lf\n", i, j, -testFitness_IN);
            freeChromosome(bestChromo);  

            // Run CGPDE-OUT (valid for T and V versions)
            struct chromosome ** populationChromos = runCGPDE_OUT(params, trainingData, validationData, numGens_OUT, &seed);    

            // >>> CGPDE-OUT-T (typeCGPDE = 2)
            bestChromo = getBestDEChromosome(params, populationChromos, validationData, 2, &seed);
            setChromosomeFitness(params, bestChromo, testingData);
            double testFitness_OUT_T = getChromosomeFitness(bestChromo);  
            printf("CGPDE-OUT-T\t%d\t%d\t%.4lf\n", i, j, -testFitness_OUT_T);
            freeChromosome(bestChromo); 
            
            // >>> CGPDE-OUT-V (typeCGPDE = 3)
            bestChromo = getBestDEChromosome(params, populationChromos, validationData, 3, &seed);
            setChromosomeFitness(params, bestChromo, testingData);
            double testFitness_OUT_V = getChromosomeFitness(bestChromo);  
            printf("CGPDE-OUT-V\t%d\t%d\t%.4lf\n", i, j, -testFitness_OUT_V);
            freeChromosome(bestChromo);            

            // Save the results
            #pragma omp critical
            {
                fprintf(f_CGP, "%d,\t%d,\t%.4f\n", i, j, -testFitness_CGP);
                fprintf(f_IN, "%d,\t%d,\t%.4f\n", i, j, -testFitness_IN);
                fprintf(f_OUT_T, "%d,\t%d,\t%.4f\n", i, j, -testFitness_OUT_T);
                fprintf(f_OUT_V, "%d,\t%d,\t%.4f\n", i, j, -testFitness_OUT_V);
            }

            // Clear the chromosomes used by CGPDE-OUT versions 
            int p;
            for (p = 0; p < NP_OUT; p++) 
            {
                freeChromosome(populationChromos[p]);
            }
            free(populationChromos);

            // Clear training, validation, and testing sets
            freeDataSet(trainingData);
            freeDataSet(validationData);
            freeDataSet(testingData);
            free(training_index);
            free(validation_index);
        }

        // Clear folds and reducedData
        int k;
        for(k = 0; k < 10; k++)
        {
            freeDataSet(folds[k]);
        }
        free(folds);	    
	freeDataSet(reducedData);	    
    }
	
    // Free the remaining variables
    freeDataSet(mainData);  
    freeParameters(params);
    fclose(f_CGP);
    fclose(f_IN);
    fclose(f_OUT_T);
    fclose(f_OUT_V);

    printf("\n* * * * * END * * * * *\n"); 

    return 0;
}

/* 
    Accuracy: the proportion of correctly classified instances
    The output node that presents the higher value is defined as the class of the instance
    e.g. Consider a chromosome with 3 output nodes, their final values are:
    Output1: 0.25 | Output2: 0.34 | Output3: 0.09
    As Output2 presents the larger value, the instance is labeled as Class #2
    
    Here, we aim to minimize -(accuracy), which is equivalent to maximize +(accuracy)
*/
double accuracy(struct parameters *params, struct chromosome *chromo, struct dataSet *data)
{
    int i,j;
    int accuracy = 0;

    if(getNumChromosomeInputs(chromo) != getNumDataSetInputs(data))
    {
        printf("Error: the number of chromosome inputs must match the number of inputs specified in the dataSet.\n");
        printf("Terminating.\n");
        exit(0);
    }

    if(getNumChromosomeOutputs(chromo) != getNumDataSetOutputs(data))
    {
        printf("Error: the number of chromosome outputs must match the number of outputs specified in the dataSet.\n");
        printf("Terminating.\n");
        exit(0);
    }

    for(i = 0; i < getNumDataSetSamples(data); i++)
    {
        executeChromosome(chromo, getDataSetSampleInputs(data, i));

        double max_predicted = -DBL_MAX;
        int predicted_class = 0;
        int correct_class = 0;

        for(j = 0; j < getNumChromosomeOutputs(chromo); j++)
        {
            double current_prediction = getChromosomeOutput(chromo,j);
            
            if(current_prediction > max_predicted)
            {
            	max_predicted = current_prediction;
            	predicted_class = j;
            }

            if(getDataSetSampleOutput(data,i,j) == 1.0)
            {
            	correct_class = j;
            }            
        }

        if(predicted_class == correct_class)
        {
        	accuracy++;
        }    
    }

    return -accuracy / (double)getNumDataSetSamples(data);
}
