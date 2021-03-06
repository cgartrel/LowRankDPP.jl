using Serialization, DataStructures, Random, LinearAlgebra

export computeLogLikelihood, computeGradient, doStochasticGradientAscent,
       doDPPLearningSparseVectorData, isConvergedLogLikelihood

# Computes the log-likelihood for the low-rank DPP with parameters paramsMatrix
function computeLogLikelihood(paramsMatrix, trainingInstances, numTrainingInstances,
                              numItems, numItemTraits, lambdaVec = 0, alpha = 0,
                              useDual = true)
  itemTraitMatrixInstance = Matrix{Float64}
  lMatrixTrainingInstance = Matrix{Float64}

  # Compute first term of log-likelihood
  sumDetTrainingInstances = 0.0
  for i = 1:numTrainingInstances
    @inbounds itemTraitMatrixInstance = paramsMatrix[trainingInstances[i], :]
    lMatrixTrainingInstance = itemTraitMatrixInstance * itemTraitMatrixInstance'

    detLMatrixTrainingInstance = det(lMatrixTrainingInstance)
    result = 0.0
    if detLMatrixTrainingInstance > 0
      result = log(detLMatrixTrainingInstance)
    end

    sumDetTrainingInstances += result
  end
  # First term of log-likelihood
  firstTerm = sumDetTrainingInstances

  # Compute second term of log-likelihood.  We optionally use the dual
  # representation of a DPP, to avoid computing the determinant of a potentially
  # large L matrix
  tItemTraitMatTimesItemTraitMat = 0
  if useDual
    tItemTraitMatTimesItemTraitMat = paramsMatrix' * paramsMatrix
  else
    tItemTraitMatTimesItemTraitMat = paramsMatrix * paramsMatrix'
  end

  detParams = det(tItemTraitMatTimesItemTraitMat + I)
  if detParams < 0
    # Fix for possible numerical stability issues
    detParams = abs(detParams)
  end
  logDetParams = log(detParams)

  # Second term of log-likelihood
  secondTerm = numTrainingInstances * logDetParams

  # Third term of log-likelihood, which is the regularization term
  thirdTerm = 0.0
  if lambdaVec != 0
    for i = 1:numItems
      thirdTerm += lambdaVec[i] * vecnorm(paramsMatrix[i, :]) ^ 2
    end

    thirdTerm *= 0.5

    if alpha != 0
      thirdTerm *= alpha
    end
  end

  # Full log-likelihood
  logLikelihood = firstTerm - secondTerm - thirdTerm

  return logLikelihood
end

# Computes the gradient of the log-likelihood for the low-rank DPP with
# parameters paramsMatrix. The gradient for each parameter (element of
# paramsMatrix) is computed with respect to that parameter.  We exploit the
# structure of the matrices and derivatives to speed up the computation of the
# gradient.
# If usePinv is set to true, we use the pseudo-inverse (pinv()) to  compute the
# gradient, instead of the inverse (inv()), which may help reduce numerical
# errors when paramsMatrix is rank deficient.  Note that pinv() is slower
# than inv() in Julia.
function computeGradient(paramsMatrix::Matrix{Float64}, trainingInstances::Vector{Vector{Int64}},
                         numTrainingInstances, numItems, numItemTraits,
                         lambdaVec::Vector{Float64} = zeros(numItems), alpha = 0,
                         paramsMatrixRowIdicesToTrainingInstanceRowIndices::Matrix{Int64} =
                         fill(0, numTrainingInstances, numItems) ;
                         usePinv = false)
  sumTraceTrainingInstances = 0.0
  paramsMatrixNumRows = numItems
  paramsMatrixNumCols = numItemTraits
  paramsMatrixGradient = zeros(Float64, numItems, numItemTraits)
  itemTraitMatrixInstance = Matrix{Float64}
  lMatrixTrainingInstance = Matrix{Float64}
  lMatrixTrainingInstanceInverse = Matrix{Float64}
  traceFirstTerm = 0.0
  sumTraceFirstTerm = 0.0
  traceSecondTerm = 0.0

  # Precompute items used in first term of gradient:
  # itemTraitMatrixInstance, lMatrixTrainingInstanceInverse, and
  # numTrainingInstanceItems for each training instance
  numTrainingInstances = Int(numTrainingInstances)
  itemTraitMatrixTrainingInstanceVec = Array{Matrix{Float64}}(undef, numTrainingInstances)
  lMatrixTrainingInstanceInverseVec = Array{Matrix{Float64}}(undef, numTrainingInstances)
  numTrainingInstanceItemsVec = Array{Int}(undef, numTrainingInstances)
  trainingInstanceRowIndex = 0
  numTrainingInstanceItems = 0
  itemNotPresentInTrainingInstance = false
  if paramsMatrixRowIdicesToTrainingInstanceRowIndices == fill(0, numTrainingInstances, numItems)
    paramsMatrixRowIdicesToTrainingInstanceRowIndices =
      buildMapTrainingInstanceRowIndices(trainingInstances, numTrainingInstances, numItems)
  end
  for trainingInstanceIndex = 1:numTrainingInstances
    @inbounds itemTraitMatrixInstance = paramsMatrix[trainingInstances[trainingInstanceIndex], :]
    @inbounds itemTraitMatrixTrainingInstanceVec[trainingInstanceIndex] = itemTraitMatrixInstance

    lMatrixTrainingInstance = itemTraitMatrixInstance * itemTraitMatrixInstance'
    if usePinv
      @inbounds lMatrixTrainingInstanceInverseVec[trainingInstanceIndex] = pinv(lMatrixTrainingInstance)
    else
      @inbounds lMatrixTrainingInstanceInverseVec[trainingInstanceIndex] = inv(lMatrixTrainingInstance)
    end

    @inbounds numTrainingInstanceItemsVec[trainingInstanceIndex] = size(itemTraitMatrixInstance, 1)
  end

  # Precompute items used in second term of gradient
  tItemTraitMatTimesItemTraitMat = paramsMatrix' * paramsMatrix
  numRowsTItemTraitMatTimesItemTraitMat = size(tItemTraitMatTimesItemTraitMat, 1)
  dualMat = zeros(numRowsTItemTraitMatTimesItemTraitMat, numRowsTItemTraitMatTimesItemTraitMat)
  if usePinv
    dualMat = paramsMatrix *
      pinv(I + tItemTraitMatTimesItemTraitMat) * paramsMatrix'
  else
    dualMat = paramsMatrix *
      inv(I + tItemTraitMatTimesItemTraitMat) * paramsMatrix'
  end

  identMinusDualMat = I - dualMat

  # Iterate over each element of currParamsMatrix to compute gradient for each element
  for paramsMatrixColIndex = 1:numItemTraits
    for paramsMatrixRowIndex = 1:numItems
      sumTraceTrainingInstances = 0.0
      for trainingInstanceIndex = 1:numTrainingInstances
        @inbounds numTrainingInstanceItems = numTrainingInstanceItemsVec[trainingInstanceIndex]

        @inbounds itemTraitMatrixInstance = itemTraitMatrixTrainingInstanceVec[trainingInstanceIndex]
        @inbounds lMatrixTrainingInstanceInverse = lMatrixTrainingInstanceInverseVec[trainingInstanceIndex]

        # Map paramsMatrixRowIndex to the proper row index in lMatrixTrainingInstanceInverse
        @inbounds trainingInstanceRowIndex = paramsMatrixRowIdicesToTrainingInstanceRowIndices[trainingInstanceIndex, paramsMatrixRowIndex]
        if trainingInstanceRowIndex != 0
          itemNotPresentInTrainingInstance = false
        else
          itemNotPresentInTrainingInstance = true
        end

        # Compute first term of gradient
        traceFirstTerm = 0.0
        if itemNotPresentInTrainingInstance
          # The lMatrixTrainingInstance derivative goes to 0 when we encounter an item
          # not present in the training instance
          traceFirstTerm = 0.0
        else
          sumTraceFirstTerm = 0.0
          @simd for i = 1:numTrainingInstanceItems
            @inbounds sumTraceFirstTerm += lMatrixTrainingInstanceInverse[i, trainingInstanceRowIndex] *
              itemTraitMatrixInstance[i, paramsMatrixColIndex]
          end

          @inbounds @views traceFirstTerm = dot(lMatrixTrainingInstanceInverse[trainingInstanceRowIndex, :],
            itemTraitMatrixInstance[:, paramsMatrixColIndex]) + sumTraceFirstTerm
        end

        sumTraceTrainingInstances += traceFirstTerm
      end
      # First term of gradient
      firstTerm = sumTraceTrainingInstances

      # Compute second term of gradient
      traceSecondTerm = 0.0
      sumTraceSecondTerm = 0.0
      @simd for i = 1:paramsMatrixNumRows
        @inbounds sumTraceSecondTerm += identMinusDualMat[i, paramsMatrixRowIndex] *
          paramsMatrix[i, paramsMatrixColIndex]
      end

      @inbounds @views traceSecondTerm = dot(identMinusDualMat[paramsMatrixRowIndex, :], paramsMatrix[:, paramsMatrixColIndex]) + sumTraceSecondTerm

      # Second term of gradient
      secondTerm = numTrainingInstances * traceSecondTerm

      # Compute the third term of the gradient, which is the regularization term
      thirdTerm = 0.0
      if lambdaVec != 0
        thirdTerm = lambdaVec[paramsMatrixRowIndex] *
          paramsMatrix[paramsMatrixRowIndex, paramsMatrixColIndex]

        if alpha != 0
          thirdTerm *= alpha
        end
      end

      # Full gradient
      @inbounds paramsMatrixGradient[paramsMatrixRowIndex, paramsMatrixColIndex] =
        firstTerm - secondTerm - thirdTerm
    end
  end

  return paramsMatrixGradient
end

# Performs stochastic gradient ascent to learn low-rank DPP kernel parameters.
# If usePinv is set to true, we use the pseudo-inverse (pinv()) to  compute the
# gradient, instead of the inverse (inv()), which may help reduce numerical
# errors when paramsMatrix is rank deficient.  Note that pinv() is slower
# than inv() in Julia.
function doStochasticGradientAscent(trainingInstances, numTrainingInstances, numItems,
                          numItemTraits, testInstances, numTestInstances, lambdaVec, alpha,
                          validationInstances = fill(Vector{Int}(), 0), numValidationInstances = 0,
                          initialParamsMatrix = rand(numItems, numItemTraits) + ones(numItems, numItemTraits),
                          usePinv = false; verbose = true, maxIters = Inf, minibatchSize = 1000)
  gradient = zeros(numItems, numItemTraits)
  paramsMatrixPrev = zeros(numItems, numItemTraits)

  paramsMatrix = initialParamsMatrix

  epsFixed = 0.5e-2
  epsInitialDecay = 1.0e-5 # For Amazon registry apparel dataset
  eps = epsFixed

  # Number of iterations for which eps is kept fixed
  numIterationsFixedEps = 50

  betaMomentum = 0.95

  numIterationsCompleted = 0

  delta = zeros(numItems, numItemTraits)

  # Number of training instances to process per minibatch
  # minibatchSize = 1000

  currTrainingInstanceIndex = 1

  shuffle!(trainingInstances)

  # Set aside one percent of trainingInstances for use as a validation set, for
  # assessing convergence, if a validation set is not explicitly provided
  avgValidationLogLikelihood = 0
  if numValidationInstances == 0
    validationSetSizePercent = 0.01
    numValidationInstances = convert(Int, round(numTrainingInstances * validationSetSizePercent));
    validationInstances = fill(Vector{Int}(), numValidationInstances)
    validationInstances = trainingInstances[1:numValidationInstances]
    trainingInstances = trainingInstances[(numValidationInstances + 1):numTrainingInstances]
    numTrainingInstances = numTrainingInstances - numValidationInstances
  end

  startTime = time()

  validationLogLike = computeLogLikelihood(paramsMatrix, validationInstances,
    numValidationInstances, numItems, 0)

  # Run stochastic gradient ascent until convergence, or until maxIters
  # iterations have been completed
  while numIterationsCompleted < maxIters
    # Get current minibatch
    numTrainingInstancesInMinibatch = minibatchSize
    if currTrainingInstanceIndex + minibatchSize > numTrainingInstances
      numTrainingInstancesInMinibatch = numTrainingInstances - currTrainingInstanceIndex
    end
    minibatchTrainingInstances =
      trainingInstances[currTrainingInstanceIndex:(currTrainingInstanceIndex + numTrainingInstancesInMinibatch)]

    paramsMatrixPrev = paramsMatrix

    if verbose
      @time gradient = computeGradient(paramsMatrix + betaMomentum * delta,
        minibatchTrainingInstances, numTrainingInstancesInMinibatch, numItems,
        numItemTraits, lambdaVec, alpha)
    else
      gradient = computeGradient(paramsMatrix + betaMomentum * delta,
        minibatchTrainingInstances, numTrainingInstancesInMinibatch, numItems,
        numItemTraits, lambdaVec, alpha)
    end


    # Use momentum when computing the update
    delta = betaMomentum * delta + (1.0 - betaMomentum) * eps * gradient
    paramsMatrix = paramsMatrix + delta

    if verbose
      @time avgTrainingLogLikelihood = computeLogLikelihood(paramsMatrix,
        trainingInstances, numTrainingInstances,
        numItems, numItemTraits) / numTrainingInstances
      @time avgValidationLogLikelihood = computeLogLikelihood(paramsMatrix,
        validationInstances, numValidationInstances, numItems,
        numItemTraits) / numValidationInstances
      @time avgTestLogLikelihood = computeLogLikelihood(paramsMatrix,
        testInstances, numTestInstances, numItems, numItemTraits) / numTestInstances

      println("avgTrainingLogLikelihood: $avgTrainingLogLikelihood")
      println("avgValidationLogLikelihood: $avgValidationLogLikelihood")
      println("avgTestLogLikelihood: $avgTestLogLikelihood")
    end

    numIterationsCompleted += 1

    if verbose
      if numIterationsCompleted % 1 == 0
        println("alpha: $alpha")
        println("Completed $numIterationsCompleted stochastic gradient ascent iterations")
      end
    end

    if numIterationsCompleted % 100 == 0
      # Save paramsMatrix to disk, so that we can reuse it later
      # save("learnedDPPParamsMatrix-k$numItemTraits-lambdaPop$alpha-$numIterationsCompleted.jld", "learnedParamsMatrix", paramsMatrix)
    end


    prevValidationLogLike = validationLogLike
    validationLogLike = computeLogLikelihood(paramsMatrix, validationInstances,
      numValidationInstances, numItems, 0)
    if isConvergedLogLikelihood(prevValidationLogLike, validationLogLike,
      1.0e-5, "validation", verbose = verbose)
      break
    end

    # Anneal (gradually lower) the learning rate
    if numIterationsCompleted >= numIterationsFixedEps
      betaMomentum = 0.0
      eps = epsInitialDecay / (1 + numIterationsCompleted / numIterationsFixedEps)

      if verbose
        println("Reduced eps: $eps")
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

  endTime = time() - startTime

  return paramsMatrix, avgValidationLogLikelihood, endTime
end

# Builds a matrix containing a map of paramsMatrixRowIndex to the proper row index in
# lMatrixTrainingInstanceInverse.  Used for computing gradient.
function buildMapTrainingInstanceRowIndices(trainingInstances, numTrainingInstances, numItems)
  numTrainingInstancesProcessed = 0

  paramsMatrixRowIdicesToTrainingInstanceRowIndicesMat = fill(0, numTrainingInstances, numItems)

  for trainingInstanceIndex = 1:numTrainingInstances
    trainingInstanceItems = trainingInstances[trainingInstanceIndex]

    for trainingInstanceRowIndex = 1:length(trainingInstanceItems)
      paramsMatrixRowIndex = trainingInstanceItems[trainingInstanceRowIndex]

      paramsMatrixRowIdicesToTrainingInstanceRowIndicesMat[trainingInstanceIndex, paramsMatrixRowIndex] = trainingInstanceRowIndex
    end
  end

  return paramsMatrixRowIdicesToTrainingInstanceRowIndicesMat
end

# Determine if we have reached convergence, based on the log likelihood
# of the V (parameter) matrix.
function isConvergedLogLikelihood(prevLogLike, newLogLike, eps, instanceTypeName;
                                  verbose = true)
  # eps = 1.0e-7
  relativeChange = abs(newLogLike - prevLogLike) / abs(prevLogLike)
  relativeChangeSign = ""
  if newLogLike < prevLogLike
    relativeChangeSign = "-"
  end

  if verbose
    println("\t relativeChange in $instanceTypeName log likelihood: $relativeChangeSign$relativeChange")
  end

  if relativeChange <= eps
    if verbose
      println("relativeChange in $instanceTypeName log likelihood: $relativeChangeSign$relativeChange, converged")
    end

    return true
  else
    return false
  end
end

# Performs low-rank DPP model learning, with item-popularity regularization, on
# a dataset in sparse vector format.
function doDPPLearningSparseVectorData(trainingBasketsDictFileName, trainingBasketsDictObjectName,
                                       testBasketsDictFileName, testBasketsDictObjectName,
                                       learnedModelOutputDirName, numItemTraits,
                                       alpha)
  println("Starting doDPPLearningSparseVectorData()")

  Random.seed!(1234)

  # cd("$(homedir())\\Belgian-retail-supermarket")

  # Load training data
  trainingUsersBasketsDict = open(deserialize, trainingBasketsDictFileName);
  println("Loaded $trainingBasketsDictFileName")

  # Load test data
  testUsersBasketsDict = open(deserialize, testBasketsDictFileName);
  println("Loaded $testBasketsDictFileName")

  # Build set of training instances
  numTrainingInstances = length(collect(keys(trainingUsersBasketsDict)))
  trainingInstances = fill(Vector{Int}(), numTrainingInstances)
  trainingInstanceIndex = 1
  numItems = 0
  for trainingInstanceUserId in collect(keys(trainingUsersBasketsDict))
    trainingInstanceItems = deepcopy(trainingUsersBasketsDict[trainingInstanceUserId].basketItems)
    heldOutItem = trainingUsersBasketsDict[trainingInstanceUserId].heldOutItem
    push!(trainingInstanceItems, heldOutItem)

    # Remove duplicates from trainingInstanceItems
    trainingInstanceItems = collect(Set(trainingInstanceItems))

    trainingInstances[trainingInstanceIndex] = trainingInstanceItems

    trainingInstanceIndex += 1

    numItems = trainingUsersBasketsDict[trainingInstanceUserId].numItemsInCatalog
  end

  # Build set of test instances
  numTestInstances = length(collect(keys(testUsersBasketsDict)))
  testInstances = fill(Vector{Int}(), numTestInstances)
  testInstanceIndex = 1
  for testInstanceUserId in collect(keys(testUsersBasketsDict))
    testInstanceItems = deepcopy(testUsersBasketsDict[testInstanceUserId].basketItems)
    heldOutItem = testUsersBasketsDict[testInstanceUserId].heldOutItem
    push!(testInstanceItems, heldOutItem)

    # Remove duplicates from testInstanceItems
    testInstanceItems = collect(Set(testInstanceItems))

    testInstances[testInstanceIndex] = testInstanceItems

    testInstanceIndex += 1
  end

  # Compute ordered item counts in training data, by descending popularity
  orderedItemCounts = PriorityQueue(Dict{Int, Int}(), Base.Order.Reverse)
  for i = 1:numTrainingInstances
    trainingInstance = trainingInstances[i]

    for itemId in trainingInstance
      if haskey(orderedItemCounts, itemId)
        orderedItemCounts[itemId] += 1
      else
        # Create a new entry for this item
        orderedItemCounts[itemId] = 1
      end
    end
  end

  # Build lambdaVec, which is a vector of per-item lambda regularization
  # parameters that are inversely proportional to item counts (popularity)
  lambdaVec = zeros(numItems)
  for i = 1:numItems
    if haskey(orderedItemCounts, i)
      lambdaVec[i] = 1 / orderedItemCounts[i]
    else
      # Item is not present in training data, so assume that it has the
      # lowest popularity value available
      lambdaVec[i] = 1
    end
  end

  println("Beginning stochastic gradient ascent...")

  # Perform stochastic gradient ascent to learn the DPP kernel parameters (item trait vectors)
  paramsMatrix, avgValidationLogLikelihood, runTime =
    doStochasticGradientAscent(trainingInstances, numTrainingInstances,
      numItems, numItemTraits, testInstances, numTestInstances, lambdaVec, alpha,
      verbose = true)

  # Compute average log-likelihood on training and test data
  avgTrainingLogLikelihood =
    computeLogLikelihood(paramsMatrix, trainingInstances, numTrainingInstances,
                         numItems, numItemTraits) / numTrainingInstances
  avgTestLogLikelihood =
    computeLogLikelihood(paramsMatrix, testInstances, numTestInstances,
                         numItems, numItemTraits) / numTestInstances
  percentDiffTraingTestLikelihood =
    (abs(avgTrainingLogLikelihood - avgTestLogLikelihood) / abs(avgTrainingLogLikelihood)) * 100

  println("Number of item trait dimensions: $numItemTraits")
  println("Final avg training log-likelihood: $avgTrainingLogLikelihood")
  println("Final avg test log-likelihood: $avgTestLogLikelihood")
  println("Diff between avg training and test log-likelihood: $percentDiffTraingTestLikelihood%")

  # Save paramsMatrix to disk, so that we can reuse it later
  if !isdir(learnedModelOutputDirName)
    mkdir(learnedModelOutputDirName)
  end
  open(f -> serialize(f, paramsMatrix),
    "$learnedModelOutputDirName/learnedDPPParamsMatrix-k$numItemTraits-lambdaPop$alpha.jls", "w");
end
