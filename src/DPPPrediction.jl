using Serialization, Random, LinearAlgebra

export TestResult, BasketSizeResult, conditionDPPOnItemsObservedLowRank,
       computeNextSingletonProbsConditionalKDPPLowRank!,
       computeNextSingletonNormalizedProbsConditionalKDPPLowRank!,
       computePredictionsSparseVectorData, computePredictionsForMCMCSamples,
       conditionDPPOnItemsObservedLowRankDual,
       computeNextSingletonProbsConditionalKDPPLowRankDual!

mutable struct TestResult
  testInstanceId::Int
  actualNextItemId::Int
  predictedRankActualNextItem::Int
  nextItemsProbs::Dict{Int, Float64}
  itemsInBasket::Set{Int}
end

mutable struct BasketSizeResult
  basketSize::Int
  numBasketInstances::Int
  sumPredictedRanks::Int
  predictedRanks::Vector{Int}
end

# Returns the L Matrix for a DPP conditioned on the event that all of the items
# (elements) in the set itemsObserved are observed, for the DPP with the
# specified lMatrix.  itemsObserved is an array that contains the item ids
# (indices) for the observed items.  The conditional DPP L matrix is computed
# using the low-rank representation of the L matrix, which significantly improves
# runtime performance.
function conditionDPPOnItemsObservedLowRank(itemTraitMatrix, itemsObserved)
  numAllItems = size(itemTraitMatrix, 1)
  allItemsSet = Set(collect(1:numAllItems))
  itemsObservedSet = Set(itemsObserved)
  numTraitDimensions = size(itemTraitMatrix, 2)
  allItemsNotInObserved = collect(setdiff(allItemsSet, itemsObservedSet))

  itemTraitMatrixAllItemsNotInObserved = itemTraitMatrix[allItemsNotInObserved, :]
  itemTraitMatrixItemsObserved = itemTraitMatrix[itemsObserved, :]

  # zMatrixConditionedOnItemsObserved is a projection matrix
  zMatrixConditionedOnItemsObserved = I -
    itemTraitMatrixItemsObserved' * inv(itemTraitMatrixItemsObserved * itemTraitMatrixItemsObserved') *
    itemTraitMatrixItemsObserved

  itemTraitMatrixConditionedOnItemsObserved =
    itemTraitMatrixAllItemsNotInObserved * zMatrixConditionedOnItemsObserved

  lMatrixConditionedOnItemsObserved =
    itemTraitMatrixConditionedOnItemsObserved * itemTraitMatrixConditionedOnItemsObserved'

  # Create a dict mapping original itemsIds in allItemsNotInObserved to row/col
  # indices in lMatrixConditionedOnItemsObserved
  itemIdsToLMatrixItemsObservedRowColIndices = Dict{Int, Int}()
  lMatrixItemsObservedRowIndex = 1
  for itemId in allItemsNotInObserved
    itemIdsToLMatrixItemsObservedRowColIndices[itemId] = lMatrixItemsObservedRowIndex

    lMatrixItemsObservedRowIndex += 1
  end

  return lMatrixConditionedOnItemsObserved, itemTraitMatrixConditionedOnItemsObserved,
    itemIdsToLMatrixItemsObservedRowColIndices
end

# Returns the L Matrix for a DPP conditioned on the event that all of the items
# (elements) in the set itemsObserved are observed, for the DPP with the
# specified L matrix.  itemsObserved is an array that contains the item ids
# (indices) for the observed items.  The conditional DPP L matrix is computed
# using the dual low-rank representation of the L matrix, which significantly improves
# runtime performance for large item catalogs.
function conditionDPPOnItemsObservedLowRankDual(itemTraitMatrix, itemsObserved)
  numAllItems = size(itemTraitMatrix, 1)
  allItemsSet = Set(collect(1:numAllItems))
  itemsObservedSet = Set(itemsObserved)
  numTraitDimensions = size(itemTraitMatrix, 2)
  allItemsNotInObserved = collect(setdiff(allItemsSet, itemsObservedSet))

  itemTraitMatrixB = itemTraitMatrix'
  cMatrix = itemTraitMatrixB * itemTraitMatrixB'

  itemTraitMatrixBAllItemsNotInObserved = itemTraitMatrixB[:, allItemsNotInObserved]
  itemTraitMatrixBItemsObserved = itemTraitMatrixB[:, itemsObserved]

  # zMatrixConditionedOnItemsObserved is a projection matrix
  zMatrixConditionedOnItemsObserved = I -
    itemTraitMatrixBItemsObserved * inv(itemTraitMatrixBItemsObserved' * itemTraitMatrixBItemsObserved) *
    itemTraitMatrixBItemsObserved'

  cMatrixConditionedOnItemsObserved = zMatrixConditionedOnItemsObserved *
    cMatrix * zMatrixConditionedOnItemsObserved

  # Create a dict mapping original itemsIds in allItemsNotInObserved to row/col
  # indices in kMatrixConditionedOnItemsObserved
  itemIdsToKMatrixItemsObservedRowColIndices = Dict{Int, Int}()
  kMatrixItemsObservedRowIndex = 1
  for itemId in allItemsNotInObserved
    itemIdsToKMatrixItemsObservedRowColIndices[itemId] = kMatrixItemsObservedRowIndex

    kMatrixItemsObservedRowIndex += 1
  end

  itemTraitMatrixBConditionedOnItemsObserved = zMatrixConditionedOnItemsObserved *
    itemTraitMatrixBAllItemsNotInObserved

  return cMatrixConditionedOnItemsObserved, itemTraitMatrixBConditionedOnItemsObserved,
    itemIdsToKMatrixItemsObservedRowColIndices
end

# Returns the unnormalized probabilities for observing each item in nextItems, given
# the DPP with the specified itemTraitMatrix and a set of observed items.
# Uses a conditional k-DPP to compute these probabilities.  The
# conditional DPP L matrix is computed using the low-rank representation
# of the L matrix, which significantly improves runtime performance.
function computeNextSingletonProbsConditionalKDPPLowRank!(itemTraitMatrix, itemsObserved,
                                                          nextItems, numAllItems, nextItemsProbs)
  lMatrixItemsObserved, itemTraitMatrixItemsObserved, itemIdsToLMatrixItemsObservedRowColIndices =
    conditionDPPOnItemsObservedLowRank(itemTraitMatrix, itemsObserved)

  for nextItemId in nextItems
    # Compute the probability of observing nextItemId, using the k-DPP defined
    # by lMatrixItemsObserved

    # The conditional probability for a singleton item is simply the corresponding
    # diagonal element of lMatrixItemsObserved
    lMatrixItemsObservedRowColIndex = itemIdsToLMatrixItemsObservedRowColIndices[nextItemId]
    nextItemsProbs[nextItemId] = lMatrixItemsObserved[lMatrixItemsObservedRowColIndex,
                                                      lMatrixItemsObservedRowColIndex]
  end

  return
end

# Returns the unnormalized probabilities for observing each item in nextItems, given
# the DPP with the specified itemTraitMatrix and a set of observed items.
# Uses a conditional k-DPP to compute these probabilities.  The
# conditional DPP L matrix is computed using the dual low-rank representation
# of the L matrix, which significantly improves runtime performance.
function computeNextSingletonProbsConditionalKDPPLowRankDual!(
  itemTraitMatrix, itemsObserved, nextItems, numAllItems, nextItemsProbs)

  cMatrixConditionedOnItemsObserved, itemTraitMatrixBConditionedOnItemsObserved,
    itemIdsToKMatrixItemsObservedRowColIndices =
    conditionDPPOnItemsObservedLowRankDual(itemTraitMatrix, itemsObserved)

  eigenDecomp = eigen(Symmetric(cMatrixConditionedOnItemsObserved))
  eigenVals = eigenDecomp.values
  eigenVecs = eigenDecomp.vectors
  rankItemTraitMatrix = size(itemTraitMatrix, 2)

  # Precompute terms used in computing probability of observing nextItemIds below
  coefficient = Vector{Float64}(undef, rankItemTraitMatrix)
  for n = 1:rankItemTraitMatrix
    coefficient[n] = (eigenVals[n] / (eigenVals[n] + 1)) * (1 / eigenVals[n])
  end

  itemsObservedRowColIndex = 0
  kMatrixNextItemIdEntry = 0
  for nextItemId in nextItems
    # Compute the probability of observing nextItemId, by computing each diagonal
    # entry of K (the marginal kernel) conditioned on observing itemsObserved
    itemsObservedRowColIndex = itemIdsToKMatrixItemsObservedRowColIndices[nextItemId]
    kMatrixNextItemIdEntry = 0
    for n = 1:rankItemTraitMatrix
      @inbounds kMatrixNextItemIdEntry += coefficient[n] *
        (itemTraitMatrixBConditionedOnItemsObserved[:, itemsObservedRowColIndex]'
          * eigenVecs[:, n]) ^ 2

      nextItemsProbs[nextItemId] = kMatrixNextItemIdEntry
    end
  end

  return
end

# Returns the normalized probabilities for observing each item in nextItems,
# give the DPP with the specified itemTraitMatrix and a set of observed items.
# Uses a conditional k-DPP to compute these probabilities.  The
# conditional DPP L matrix is computed using the low-rank representation
# of the L matrix, which significantly improves runtime performance.
function computeNextSingletonNormalizedProbsConditionalKDPPLowRank!(
  itemTraitMatrix, itemsObserved, nextItems, numAllItems, nextItemsProbs)

  lMatrixItemsObserved, itemTraitMatrixItemsObserved, itemIdsToLMatrixItemsObservedRowColIndices =
    conditionDPPOnItemsObservedLowRank(itemTraitMatrix, itemsObserved)

  # We use the dual representation of lMatrixItemsObserved for computing the
  # normalizer, to avoid computing the eigenvalues of a potentially large L
  # matrix.  The nonzero eigenvalues of lMatrixItemsObserved and
  # lMatrixItemsObservedDual are identical.
  lMatrixItemsObservedDual = itemTraitMatrixItemsObserved' * itemTraitMatrixItemsObserved

  # Normalizer is the first elementary symmetric polynomial on the eigenvalues
  # of lMatrixItemsObserved.  We only need to compute k eigenvalues, where
  # k is the rank of lMatrixItemsObserved (rank of itemTraitMatrix).
  rankLMatrixItemsObserved = size(itemTraitMatrix, 2)
  numItemsObserved =  size(lMatrixItemsObserved, 1)
  eigenValsLMatrixItemsObserved = eigvals(Symmetric(lMatrixItemsObservedDual))
  firstElemSymPoly = sum(eigenValsLMatrixItemsObserved)
  normalizationConstant = firstElemSymPoly

  for nextItemId in nextItems
    # Compute the probability of observing nextItemId, using the k-DPP defined
    # by lMatrixItemsObserved

    # The conditional probability for a singleton item is simply proportional
    # to the corresponding diagonal element of lMatrixItemsObserved
    lMatrixItemsObservedRowColIndex = itemIdsToLMatrixItemsObservedRowColIndices[nextItemId]
    nextItemsProbs[nextItemId] = lMatrixItemsObserved[lMatrixItemsObservedRowColIndex,
      lMatrixItemsObservedRowColIndex] / normalizationConstant
  end

  return
end

# Computes DPP single-item basket completion predictions on a dataset in sparse vector format.
# The learned DPP model parameters (learnedDPPParamsFileName) represents a model
# learned by stochastic gradient ascent using the DPPLearning module.
# If useDual is set to true, the dual version of the low-rank DPP kernel is
# used to compute predictions, which is significantly faster for large item
# catalogs.
function computePredictionsSparseVectorData(testBasketsDictFileName,
  testBasketsDictObjectName, learnedDPPParamsFileName,
  resultsForTestInstancesDictFileName, learnedDPPParamsObjectName = "learnedParamsMatrix",
  useDual = false)
  Random.seed!(1234)

  testUsersBasketsDict = open(deserialize, testBasketsDictFileName);
  println("Loaded $testBasketsDictFileName")

  # Build set of test instances
  numTestInstances = length(collect(keys(testUsersBasketsDict)))
  testInstances = fill(Vector{Int}(), numTestInstances)
  testInstanceIndex = 1
  numItems = 0
  for testInstanceUserId in collect(keys(testUsersBasketsDict))
    testInstanceItems = deepcopy(testUsersBasketsDict[testInstanceUserId].basketItems)
    heldOutItem = testUsersBasketsDict[testInstanceUserId].heldOutItem
    push!(testInstanceItems, heldOutItem)

    testInstances[testInstanceIndex] = testInstanceItems

    testInstanceIndex += 1

    numItems = testUsersBasketsDict[testInstanceUserId].numItemsInCatalog
  end

  numDistinctTestInstances = length(Set(testInstances))
  println("Number of test instances: $(length(testInstances))")
  println("Number of distinct test instances: $numDistinctTestInstances")

  # Load serialized trained DPP model from disk
  learnedParamsMatrix = open(deserialize, learnedDPPParamsFileName);
  println("Loaded $learnedDPPParamsFileName")

  allItemsSet = Set(1:numItems)

  nextItemsProbs = Dict{Int, Float64}()
  resultsForTestInstancesDict = Dict{Vector{Int}, TestResult}()

  println("Processing testInstances 1 to $(length(testInstances))")
  # Compute next-item predictions for each test instance (basket)
  startTime = time()
  for i = 1:length(testInstances)
    testInstance = testInstances[i]

    if haskey(resultsForTestInstancesDict, testInstance)
      continue
    end

    testInstanceLength = length(testInstance)

    actualNextItem = testInstance[testInstanceLength]
    observedItemsInBasket = testInstance[1:(testInstanceLength - 1)]

    observedItemsInBasketSet = Set(observedItemsInBasket)
    nextItemsForPredictionSet = setdiff(allItemsSet, observedItemsInBasketSet)

    if useDual
      computeNextSingletonProbsConditionalKDPPLowRankDual!(learnedParamsMatrix,
        observedItemsInBasket, collect(nextItemsForPredictionSet), numItems,
        nextItemsProbs)
    else
      computeNextSingletonProbsConditionalKDPPLowRank!(learnedParamsMatrix,
        observedItemsInBasket, collect(nextItemsForPredictionSet), numItems,
        nextItemsProbs)
    end


    testResult = TestResult(i, actualNextItem, 0, deepcopy(nextItemsProbs), observedItemsInBasketSet)

    # Compute rank of actual next item in the sorted list of next-item predictions
    rankActualNextItem = length(findall(x -> x > nextItemsProbs[actualNextItem], collect(values(nextItemsProbs))))
    empty!(nextItemsProbs)

    testResult.predictedRankActualNextItem = rankActualNextItem

    resultsForTestInstancesDict[testInstance] = testResult

    if i % 100 == 0
      println("Processed $i test instances")
    end
  end
  elapsedPredictionTime = time() - startTime

  averagePredictionTimePerTestInstance = elapsedPredictionTime / numDistinctTestInstances
  println("averagePredictionTimePerTestInstance = $averagePredictionTimePerTestInstance")

  # Save resultsForTestInstancesDict
  open(f -> serialize(f, resultsForTestInstancesDict),
    "$resultsForTestInstancesDictFileName", "w");
  println("Saved $resultsForTestInstancesDictFileName")

  return resultsForTestInstancesDict
end

# Computes next-item predictions from a collection of MCMC samples.
# All but every sampleLag samples are discarded.  Setting sampleLag to 1
# indicates that every sample should be used to compute predictions.
# This function computes an MCMC approximation of the posterior predictive
# distribution for the Baysian low-rank DPP model.
function computePredictionsForMCMCSamples(testBasketsDictFileName,
  testBasketsDictObjectName, collectedMCMCSamplesDirPathName, sampleLag,
  resultsForTestInstancesDictFileName)
  learnedDPPParamsObjectName = "itemTraitMatrixSample"
  # Select MCMC samples to use for computing predictions, according to
  # contents of collectedMCMCSamplesDirPathName and sampleLag
  collectedSamplesFileNamesVec = readdir(collectedMCMCSamplesDirPathName)
  sort!(collectedSamplesFileNamesVec, by = getSampleNumberFromSampleFileName)
  numCollectedSamples = length(collectedSamplesFileNamesVec)
  numSamplesForComputingPredictions = round(Int, numCollectedSamples / sampleLag)
  samplesForComputingPredictionsFileNames = Array{String}(numSamplesForComputingPredictions)
  samplesForComputingPredictionsFileNames[1] = "$collectedMCMCSamplesDirPathName/$(collectedSamplesFileNamesVec[1])"
  samplesForComputingPredictionsFileNamesIndex = 2
  for i = 0:sampleLag:numCollectedSamples
    if (i == 0) || (i == 1)
      continue
    end

    samplesForComputingPredictionsFileNames[samplesForComputingPredictionsFileNamesIndex] =
      "$collectedMCMCSamplesDirPathName/$(collectedSamplesFileNamesVec[i])"

    samplesForComputingPredictionsFileNamesIndex += 1
  end

  for i = 1:numSamplesForComputingPredictions
    println("samplesForComputingPredictionsFileNames[$i]: $(samplesForComputingPredictionsFileNames[i])")
  end
  println("Computing predictions for $numSamplesForComputingPredictions samples")

  # Compute predictions for each MCMC sample in
  # samplesForComputingPredictionsFileNames in parallel
  predictionResultsForSamples = Array{Dict{Array{Int, 1}, TestResult}}(numSamplesForComputingPredictions)
  predictionResultsForSamples = pmap((testBasketsDictFileName, testBasketsDictObjectName,
    learnedDPPParamsFileName, resultsForTestInstancesDictFileName,
    learnedDPPParamsObjectName) ->
    computePredictionsSparseVectorData(testBasketsDictFileName, testBasketsDictObjectName,
    learnedDPPParamsFileName, resultsForTestInstancesDictFileName,
    learnedDPPParamsObjectName),
    fill(testBasketsDictFileName, numSamplesForComputingPredictions),
    fill(testBasketsDictObjectName, numSamplesForComputingPredictions),
    samplesForComputingPredictionsFileNames,
    fill(resultsForTestInstancesDictFileName, numSamplesForComputingPredictions),
    fill(learnedDPPParamsObjectName, numSamplesForComputingPredictions))

  # Save predictionResultsForSamples to separate files, one for each sample
  # @sync for i = 1:numSamplesForComputingPredictions
  #   @async save("$collectedMCMCSamplesDirPathName/predictionResultsForSamples-sampleLag-$sampleLag-collectedSample$i.jld",
  #     "predictionResultForSample", predictionResultsForSamples[i])
  #   println("Saved predictionResultsForSamples-sampleLag-$sampleLag-collectedSample$i.jld")
  # end

  println("Finding max predictive log probabilities for each test instance...")

  # Find max predictive log probabilities for each test instance, across MCMC samples
  maxLogProbsPredictionResultsForSamples = Dict{Array{Int, 1}, Dict{Int, Float64}}()
  for sampleIndex = 1:numSamplesForComputingPredictions
    for testInstance in keys(predictionResultsForSamples[1])
      nextItemsLogProbs = deepcopy(predictionResultsForSamples[sampleIndex][testInstance].nextItemsProbs)
      # Work in log prob space
      for nextItemId in keys(nextItemsLogProbs)
        nextItemsLogProbs[nextItemId] = log(nextItemsLogProbs[nextItemId])
      end

      if !haskey(maxLogProbsPredictionResultsForSamples, testInstance)
        maxLogProbsPredictionResultsForSamples[testInstance] = nextItemsLogProbs
      else
        maxNextItemsLogProbs = maxLogProbsPredictionResultsForSamples[testInstance]
        for nextItemId in keys(nextItemsLogProbs)
          if nextItemsLogProbs[nextItemId] > maxNextItemsLogProbs[nextItemId]
            # Found a larger log probability than is currently stored in
            # maxLogProbsPredictionResultsForSamples, so update the max
            maxNextItemsLogProbs[nextItemId] = nextItemsLogProbs[nextItemId]
          end
        end

        maxLogProbsPredictionResultsForSamples[testInstance] = maxNextItemsLogProbs
      end
    end
  end

  println("Computing expected values for predictive probabilities...")

  # Compute expected values for predictions in log-prob space for each MCMC sample
  expectedValuePredictionResultsForSamples = Dict{Array{Int, 1}, TestResult}()
  for testInstance in keys(predictionResultsForSamples[1])
    nextItemsProbs = predictionResultsForSamples[1][testInstance].nextItemsProbs
    expectedValueNextItemsLogProbs = Dict{Int, Float64}()
    for nextItemId in keys(nextItemsProbs)
      partialSumPredictionResultsForSamplesForNextItemId = 0
      maxLogProbNextItem = maxLogProbsPredictionResultsForSamples[testInstance][nextItemId]

      # Add probabilities in log space
      for sampleIndex = 1:numSamplesForComputingPredictions
        nextItemProb = predictionResultsForSamples[sampleIndex][testInstance].nextItemsProbs[nextItemId]

        partialSumPredictionResultsForSamplesForNextItemId +=
          exp(log(nextItemProb) - maxLogProbNextItem)
      end

      # Final step for adding probailities in log space and computing expected value
      expectedValueNextItemsLogProbs[nextItemId] =
        (maxLogProbNextItem + log(partialSumPredictionResultsForSamplesForNextItemId)) / numSamplesForComputingPredictions
    end

    # Compute rank of actual next item in the sorted list of next-item predictions
    testResult = deepcopy(predictionResultsForSamples[1][testInstance])
    rankActualNextItem = length(find(x -> x > expectedValueNextItemsLogProbs[testResult.actualNextItemId],
      collect(values(expectedValueNextItemsLogProbs))))
    testResult.predictedRankActualNextItem = rankActualNextItem
    testResult.nextItemsProbs = expectedValueNextItemsLogProbs
    expectedValuePredictionResultsForSamples[testInstance] = testResult
  end

  # Save expectedValuePredictionResultsForSamples
  save(resultsForTestInstancesDictFileName, "resultsForTestInstancesDict",
    expectedValuePredictionResultsForSamples)
  println("Saved $resultsForTestInstancesDictFileName")
end

# Returns sample number as an integer, given sample file name.
function getSampleNumberFromSampleFileName(sampleFileName)
  sampleFileNameParts = split(sampleFileName, "-")
  sampleNumberPart = sampleFileNameParts[length(sampleFileNameParts)]
  numberString = split(sampleNumberPart, ".")[1]
  return parse(Int, numberString)
end
