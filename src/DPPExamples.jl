export convertCsvToBasketsExample, dppLearningExample,
  dppLearningBayesianExample, predictionExample,
  predictionForMCMCSamplesExample, predictionMetricsExample

dataDir = "../data/Amazon-baby-registry"

# An example of converting a file in sparse CSV basket format to the data
# format expected by the DPP modules.
function convertCsvToBasketsExample()
  csvBasketDataFileName = "$dataDir/1_100_100_100_apparel_regs.csv"
  trainingBasketsDictFileName = "$dataDir/apparel-regs-training-basketsDict.jls"
  testBasketsDictFileName = "$dataDir/apparel-regs-test-basketsDict.jls"
  trainingSetSizePercent = 0.8

  convertSparseCsvToBaskets(csvBasketDataFileName, trainingBasketsDictFileName,
    testBasketsDictFileName, trainingSetSizePercent)
end

# An example of using stochastic gradient ascent for DPP learning
function dppLearningExample()
  trainingBasketsDictFileName = "$dataDir/apparel-regs-training-basketsDict.jls"
  trainingBasketsDictObjectName = "trainingBasketsDict"
  testBasketsDictFileName = "$dataDir/apparel-regs-test-basketsDict.jls"
  testBasketsDictObjectName = "testBasketsDict"
  learnedModelOutputDirName = "example-SGA-model"
  numItemTraits = 30
  alpha = 1.0

  doDPPLearningSparseVectorData(trainingBasketsDictFileName,
    trainingBasketsDictObjectName, testBasketsDictFileName,
    testBasketsDictObjectName, learnedModelOutputDirName, numItemTraits, alpha)

end

# An example of using stochastic gradient HMC for DPP learning
function dppLearningBayesianExample()
  trainingBasketsDictFileName = "$dataDir/apparel-regs-training-basketsDict.jld"
  trainingBasketsDictObjectName = "trainingBasketsDict"
  testBasketsDictFileName = "$dataDir/apparel-regs-test-basketsDict.jld"
  testBasketsDictObjectName = "testBasketsDict"
  learnedModelOutputDirName = "example-SGHMC-model"
  numItemTraits = 30

  doDPPBayesianLearningSparseVectorData(trainingBasketsDictFileName,
    trainingBasketsDictObjectName, testBasketsDictFileName,
    testBasketsDictObjectName, learnedModelOutputDirName, numItemTraits,
    runStochasticGradientHamiltonianMonteCarloSampler)
end

# An example of computing predictions using a DPP model learned by stochastic
# gradient ascent
function predictionExample()
  testBasketsDictFileName = "$dataDir/apparel-regs-test-basketsDict.jls"
  testBasketsDictObjectName = "testBasketsDict"
  numItemTraits = 30
  alpha = 1.0
  learnedDPPParamsFileName = "example-SGA-model/learnedDPPParamsMatrix-k$numItemTraits-lambdaPop$alpha.jls"
  resultsForTestInstancesDictFileName =
    "example-SGA-model/resultsForTestInstancesDict-k$numItemTraits-lambdaPop$alpha-Amazon-apparel-regs.jls"

  computePredictionsSparseVectorData(testBasketsDictFileName,
    testBasketsDictObjectName, learnedDPPParamsFileName,
    resultsForTestInstancesDictFileName)
end

# An example of computing predictions using a DPP model learned by stochastic
# gradient HMC
function predictionForMCMCSamplesExample()
  testBasketsDictFileName = "$dataDir/apparel-regs-test-basketsDict.jld"
  testBasketsDictObjectName = "testBasketsDict"
  numItemTraits = 30
  sampleLag = 1
  collectedMCMCSamplesDirPathName =
    "example-SGHMC-model/learnedDPPParams-k$numItemTraits-SGHMC-collectedSamples"
  resultsForTestInstancesDictFileName =
    "example-SGHMC-model/resultsForTestInstancesDict-k$numItemTraits-collectedSamples-SGHMC-Amazon-apparel-regs.jld"

  computePredictionsForMCMCSamples(testBasketsDictFileName,
    testBasketsDictObjectName, collectedMCMCSamplesDirPathName, sampleLag,
    resultsForTestInstancesDictFileName)
end

# An example of computing prediction metrics using DPP model prediction results
function predictionMetricsExample()
  trainingBasketsDictFileName = "$dataDir/apparel-regs-training-basketsDict.jls"
  trainingBasketsDictObjectName = "trainingBasketsDict"
  testBasketsDictFileName = "$dataDir/apparel-regs-test-basketsDict.jls"
  testBasketsDictObjectName = "testBasketsDict"
  numItemTraits = 30
  alpha = 1.0
  resultsForTestInstancesDictFileName =
    "example-SGA-model/resultsForTestInstancesDict-k$numItemTraits-lambdaPop$alpha-Amazon-apparel-regs.jls"
  # resultsForTestInstancesDictFileName =
  #   "example-SGHMC-model/resultsForTestInstancesDict-k$numItemTraits-collectedSamples-SGHMC-Amazon-apparel-regs.jls"

  computePredictionMetricsSparseVectorData(trainingBasketsDictFileName,
    trainingBasketsDictObjectName, testBasketsDictFileName,
    testBasketsDictObjectName, resultsForTestInstancesDictFileName)
end

# Optimization-based learning pipeline:
# convertCsvToBasketsExample()
# dppLearningExample()
# predictionExample()
# predictionMetricsExample()

# Bayesian learning pipeline:
# dppLearningBayesianExample()
# predictionForMCMCSamplesExample()
# predictionMetricsExample()
