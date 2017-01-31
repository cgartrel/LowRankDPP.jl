module DPPLearningBayesian

using HDF5, JLD
using Distributions
using DPPDataPreparation
using DPPLearning

export runStochasticGradientHamiltonianMonteCarloSampler,
       doDPPBayesianLearningSparseVectorData

# Computes the gradient of the log-prior density
function computePriorGradient(paramsMatrix, numItems, numItemTraits,
                              gaussianPriorPrecisions)
  paramsMatrixGradient = zeros(numItems, numItemTraits)

  for paramsMatrixColIndex = 1:numItemTraits
    for paramsMatrixRowIndex = 1:numItems
      paramsMatrixGradient[paramsMatrixRowIndex, paramsMatrixColIndex] =
        gaussianPriorPrecisions[paramsMatrixRowIndex] * paramsMatrix[paramsMatrixRowIndex, paramsMatrixColIndex]
    end
  end

  return paramsMatrixGradient
end

# Computes the gradient of the unnormalized log-posterior density
function computePosteriorGradient(paramsMatrix, trainingInstances, numTrainingInstances,
                                  numItems, numItemTraits, gaussianPriorPrecisions)
  posteriorGradient = zeros(numItems, numItemTraits)

  # Compute the gradient for the prior density, and the gradient for the likelihood function
  priorGradient = computePriorGradient(paramsMatrix, numItems, numItemTraits,
                                       gaussianPriorPrecisions)
  likelihoodGradient = computeGradient(paramsMatrix, trainingInstances,
                                       numTrainingInstances, numItems, numItemTraits)

  posteriorGradient = priorGradient + likelihoodGradient

  return posteriorGradient
end

# Runs Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) to generate samples from the posterior
function runStochasticGradientHamiltonianMonteCarloSampler(trainingInstances,
  numTrainingInstances, numItems, numItemTraits, testInstances, numTestInstances,
  learnedModelOutputDirName, initialItemTraitMatrix = -1)

  # Initialize itemTraitMatrix (V matrix)
  itemTraitMatrix = rand(numItems, numItemTraits)
  itemTraitMatrixPrev = zeros(numItems, numItemTraits)
  if initialItemTraitMatrix != -1
    itemTraitMatrix = initialItemTraitMatrix
  end

  posteriorGradient = zeros(numItems, numItemTraits)

  itemTraitMatrixSamplesSum = zeros(numItems, numItemTraits)

  # Max number of iterations (samples)
  maxNumIterations = 1000

  # Number of "burn-in" samples, which are discarded
  numBurnInSamples = 100

  # Discard all but every sampleLag samples.  Used for thinning of samples.
  sampleLag = 10

  # Step size (learning rate)
  # stepSizeLarger = 1.0e-5
  stepSizeLarger = 1.0e-4
  stepSizeIntermediate = 1.0e-6
  stepSizeSmaller = 3.0e-8
  stepSize = stepSizeLarger

  alphaMomentum = 0.01
  alphaMomentumIntermediate = 0.01
  alphaMomentumSmaller = 0.01

  numIterationsLargerStepSize = 17
  numIterationsIntermediateStepSize = 95

  # Gradient noise estimate (beta hat)
  gradientNoiseEstimate = 0.0

  # Gradient noise distribution
  gradientNoiseGaussian = Normal(0, 2 * (alphaMomentum - gradientNoiseEstimate) * stepSize)

  # Number of "leapfrog" steps
  numLeapfrogSteps = 10

  # Number of training instances to process per minibatch
  minibatchSize = 1000

  delta = zeros(numItems, numItemTraits)

  # Hyperparameters on the gamma hyperprior for each item
  gammaShapeHyperParam = sqrt(numItemTraits)
  gammaRateHyperParam = 1.0

  # Shape parameter for the gamma hyperprior for each item
  gammaShape = gammaShapeHyperParam + (numItems * numItemTraits) / 2.0

  # Number of samples collected from the posterior
  numSamplesCollected = 1

  # Create directory for storing collected samples, if it doesn't already exist
  collectedSamplesDir =
    "$learnedModelOutputDirName/learnedDPPMixtureParams-k$numItemTraits-SGHMC-collectedSamples"
  if !isdir(collectedSamplesDir)
    mkpath(collectedSamplesDir)
  end

  currTrainingInstanceIndex = 1
  shuffle!(trainingInstances)
  for iterCounter = 1:maxNumIterations
    # Get current minibatch
    numTrainingInstancesInMinibatch = minibatchSize
    if currTrainingInstanceIndex + minibatchSize > numTrainingInstances
      numTrainingInstancesInMinibatch = numTrainingInstances - currTrainingInstanceIndex
    end
    minibatchTrainingInstances =
      trainingInstances[currTrainingInstanceIndex:(currTrainingInstanceIndex + numTrainingInstancesInMinibatch)]

    # Rate parameter for the gamma hyperprior for each item
    sumMagnitudeItemVectors = 0.0
    for i = 1:numItems
      sumMagnitudeItemVectors += vecnorm(itemTraitMatrix[i, :]) ^ 2
    end
    gammaRate = gammaRateHyperParam + sumMagnitudeItemVectors / 2

    # Sample new precision for the Gaussian prior on each item from a
    # gamma hyperprior on each precision.  All items share the same sampled
    # precision parameter.
    gammaHyperprior = Gamma(gammaShape, 1 / gammaRate)
    gaussianPriorPrecisions = zeros(numItems)
    gaussianPriorPrecisionShared = rand(gammaHyperprior)
    for itemIndex = 1:numItems
      gaussianPriorPrecisions[itemIndex] = gaussianPriorPrecisionShared
    end

    # Compute new itemTraitMatrix and delta, using the SGHMC updates
    itemTraitMatrixPrev = itemTraitMatrix
    @time for j = 1:numLeapfrogSteps
      itemTraitMatrix = itemTraitMatrix + delta

      posteriorGradient =
        computePosteriorGradient(itemTraitMatrix, minibatchTrainingInstances, numTrainingInstancesInMinibatch,
                                 numItems, numItemTraits, gaussianPriorPrecisions)

      gradientNoiseSample = rand(gradientNoiseGaussian)

      delta = stepSize * posteriorGradient - alphaMomentum * delta + gradientNoiseSample
    end

    if iterCounter % 1 == 0
      println("$iterCounter iterations completed; stepSize: $stepSize; alphaMomentum: $alphaMomentum")

      avgTrainingLogLikelihood =
        computeLogLikelihood(itemTraitMatrix, trainingInstances, numTrainingInstances,
                             numItems, numItemTraits) / numTrainingInstances
      avgTestLogLikelihood =
        computeLogLikelihood(itemTraitMatrix, testInstances, numTestInstances,
                             numItems, numItemTraits) / numTestInstances
      println("\t avgTrainingLogLikelihood: $avgTrainingLogLikelihood")
      println("\t avgTestLogLikelihood: $avgTestLogLikelihood")
    end

    if iterCounter >= numIterationsLargerStepSize && iterCounter < numIterationsIntermediateStepSize
      stepSize = stepSizeIntermediate
      alphaMomentum = alphaMomentumIntermediate
    end

    if iterCounter >= numIterationsIntermediateStepSize
      stepSize = stepSizeSmaller
      alphaMomentum = alphaMomentumSmaller
    end

    if iterCounter > 2
      # Tracking training log likelihood is not a "true" measure of convergence
      # here, but provides a useful heuristic
      isConvergedLogLikelihood(itemTraitMatrix, itemTraitMatrixPrev,
        trainingInstances, numTrainingInstances, numItems, 1.0e-7, "training")
    end

    # After burn in, collect samples
    if iterCounter > numBurnInSamples
      if iterCounter % sampleLag == 0
        @async save("$collectedSamplesDir/learnedDPPMixtureParams-k$numItemTraits-SGHMC-sample-$numSamplesCollected.jld",
          "itemTraitMatrixSample", itemTraitMatrix)

        println("Collected $numSamplesCollected samples")
        numSamplesCollected += 1
      end
    end

    currTrainingInstanceIndex += numTrainingInstancesInMinibatch + 1
    if currTrainingInstanceIndex > numTrainingInstances
      # We've processed all of the training instances, so start processing the
      # training instances again from the beginning
      currTrainingInstanceIndex = 1

      shuffle!(trainingInstances)
    end
  end
end

# Performs learning for the Bayesian low-rank DPP model, using an MCMC sampler
# function, on a dataset in sparse vector format.
function doDPPBayesianLearningSparseVectorData(trainingBasketsDictFileName,
  trainingBasketsDictObjectName, testBasketsDictFileName,
  testBasketsDictObjectName, learnedModelOutputDirName, numItemTraits,
  runSamplerFunction::Function, initialItemTraitMatrxFileName = -1,
  initialItemTraitMatrxObjectName = -1)
  srand(1234)

  # Load training data
  trainingBasketsDict = load(trainingBasketsDictFileName,
                                  trainingBasketsDictObjectName)
  println("Loaded $trainingBasketsDictFileName")

  # Load test data
  testBasketsDict = load(testBasketsDictFileName,
                              testBasketsDictObjectName)
  println("Loaded $testBasketsDictFileName")

  # Build set of training instances
  numTrainingInstances = length(collect(keys(trainingBasketsDict)))
  trainingInstances = fill(Array(Int, 1), numTrainingInstances)
  trainingInstanceIndex = 1
  numItems = 0
  for trainingInstanceBasketId in collect(keys(trainingBasketsDict))
    trainingInstanceItems = deepcopy(trainingBasketsDict[trainingInstanceBasketId].basketItems)
    heldOutItem = trainingBasketsDict[trainingInstanceBasketId].heldOutItem
    push!(trainingInstanceItems, heldOutItem)

    trainingInstances[trainingInstanceIndex] = trainingInstanceItems

    trainingInstanceIndex += 1

    numItems = trainingBasketsDict[trainingInstanceBasketId].numItemsInCatalog
  end

  # Build set of test instances
  numTestInstances = length(collect(keys(testBasketsDict)))
  testInstances = fill(Array(Int, 1), numTestInstances)
  testInstanceIndex = 1
  for testInstanceBasketId in collect(keys(testBasketsDict))
    testInstanceItems = deepcopy(testBasketsDict[testInstanceBasketId].basketItems)
    heldOutItem = testBasketsDict[testInstanceBasketId].heldOutItem
    push!(testInstanceItems, heldOutItem)

    testInstances[testInstanceIndex] = testInstanceItems

    testInstanceIndex += 1
  end

  # Initialize itemTraitMatrix, if provided
  itemTraitMatrixInit = -1
  if initialItemTraitMatrxFileName != -1
    itemTraitMatrixInit = load(initialItemTraitMatrxFileName, initialItemTraitMatrxObjectName)
    println("Loaded $initialItemTraitMatrxFileName")
  end

  # Run sampler to learn the posterior on the low-rank DPP parameters (V matrix)
  runSamplerFunction(trainingInstances, numTrainingInstances, numItems,
    numItemTraits, testInstances, numTestInstances, learnedModelOutputDirName,
    itemTraitMatrixInit)
end

end
