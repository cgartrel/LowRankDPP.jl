module DPPEvaluation

using JLD
using DPPPrediction

export computePredictionMetricsSparseVectorData

# Returns a priority queue of ordered item counts, for a dataset in sparse vector format
function getOrderedItemCountsSparseVectorData(trainingBasketsDictFileName,
                                              trainingBasketsDictObjectName)

  # Load training data
  trainingUsersBasketsDict = load(trainingBasketsDictFileName,
                                  trainingBasketsDictObjectName)
  println("Loaded $trainingBasketsDictFileName")

  # Build set of training instances
  numTrainingInstances = length(collect(keys(trainingUsersBasketsDict)))
  trainingInstances = fill(Array(Int, 1), numTrainingInstances)
  trainingInstanceIndex = 1
  numItems = 0
  for trainingInstanceUserId in collect(keys(trainingUsersBasketsDict))
    trainingInstanceItems = deepcopy(trainingUsersBasketsDict[trainingInstanceUserId].basketItems)
    heldOutItem = trainingUsersBasketsDict[trainingInstanceUserId].heldOutItem
    push!(trainingInstanceItems, heldOutItem)

    trainingInstances[trainingInstanceIndex] = trainingInstanceItems

    trainingInstanceIndex += 1

    numItems = trainingUsersBasketsDict[trainingInstanceUserId].numItemsInCatalog
  end

  # Compute ordered item counts, by descending popularity
  orderedItemCounts = Collections.PriorityQueue(Dict{Int, Int}(), Base.Order.Reverse)
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

  return orderedItemCounts
end

# Compute MPR and precision metrics from DPP prediction results for a dataset in sparse vector format
function computePredictionMetricsSparseVectorData(trainingBasketsDictFileName, trainingBasketsDictObjectName,
                                                  testBasketsDictFileName, testBasketsDictObjectName,
                                                  resultsForTestInstancesDictFileName)
  resultsForTestInstancesDictObjectName = "resultsForTestInstancesDict"
  srand(1234)

  # Load test data
  testUsersBasketsDict = load(testBasketsDictFileName, testBasketsDictObjectName)
  println("Loaded $testBasketsDictFileName")

  # Build set of test instances
  numTestInstances = length(collect(keys(testUsersBasketsDict)))
  testInstances = fill(Array(Int, 1), numTestInstances)
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

  # Compute popularity-stratification weights for each item, where the weight is
  # proportional to 1 / (itemCount ^ beta), for beta = 0.5
  orderedItemCounts = getOrderedItemCountsSparseVectorData(trainingBasketsDictFileName,
                                                           trainingBasketsDictObjectName)
  sumItemCounts = sum(values(orderedItemCounts))
  itemPopWeights = Collections.PriorityQueue(Dict{Int, Float64}(), Base.Order.Reverse)
  beta = 0.5
  startingItemId = 1
  for i = startingItemId:numItems
    if haskey(orderedItemCounts, i)
      itemPopWeights[i] = 1 / (orderedItemCounts[i] ^ beta)
    else
      # If an item id is not available in orderedItemCounts, then its popularity is 0.
      itemPopWeights[i] = 0
      println("itemPopWeights[$i]: 0")
    end
  end
  # Normalize itemPopWeights, so they sum to 1
  sumItemPopWeights = sum(collect(values(itemPopWeights)))
  for i = startingItemId:numItems
    itemPopWeights[i] = itemPopWeights[i] / sumItemPopWeights
  end

  # Load DPP prediction results from disk
  resultsForTestInstancesDict = load(resultsForTestInstancesDictFileName,
                                     resultsForTestInstancesDictObjectName)
  println("Loaded $resultsForTestInstancesDictFileName")

  # Compute mean percentile rank and precision across predictions for all test instances
  shuffle!(testInstances)
  numTestInstances = length(testInstances)
  sumPredictedRanks = 0
  basketSizeResultsDict = Dict{Int, BasketSizeResult}()
  predictedRanks = Int[]
  for i = 1:numTestInstances
    testInstance = testInstances[i]

    # Get predicted rank for test instance
    testResult = resultsForTestInstancesDict[testInstance]
    sumPredictedRanks += testResult.predictedRankActualNextItem
    push!(predictedRanks, testResult.predictedRankActualNextItem)

    # Last item in the test instance is not part of the basket
    basketSize = length(testInstance) - 1
    # Update metrics for the size of the current basket
    if basketSize >= 2
      if haskey(basketSizeResultsDict, basketSize)
        basketSizeResult = basketSizeResultsDict[basketSize]
        basketSizeResult.numBasketInstances += 1
        basketSizeResult.sumPredictedRanks += testResult.predictedRankActualNextItem
        push!(basketSizeResult.predictedRanks, testResult.predictedRankActualNextItem)
        basketSizeResultsDict[basketSize] = basketSizeResult
      else
        # Create a new entry for this basket size
        basketSizeResult = BasketSizeResult(basketSize, 1, testResult.predictedRankActualNextItem,
                                            [testResult.predictedRankActualNextItem])
        basketSizeResultsDict[basketSize] = basketSizeResult
      end

      if i % 100 == 0
        println("Processed $i test instances")
      end
    end
  end

  # Compute overall mean percentile rank
  meanRank = sumPredictedRanks / numTestInstances
  meanRankPercent = meanRank / numItems
  meanPercentileRank = 100.0 - (100.0 * meanRankPercent)
  println("meanPercentileRank for all baskets: $meanPercentileRank")

  # Compute mean percentile rank for each basket size
  for basketSize in sort(collect(keys(basketSizeResultsDict)))
    basketSizeResult = basketSizeResultsDict[basketSize]
    meanRank = basketSizeResult.sumPredictedRanks / basketSizeResult.numBasketInstances
    meanRankPercent = meanRank / numItems
    meanPercentileRank = 100.0 - (100.0 * meanRankPercent)
    println("For basket size $basketSize, meanPercentileRank: $meanPercentileRank")
  end

  # Compute overall precision@1 (accuracy)
  hits = length(find(x -> x < 1, predictedRanks))
  precisionAtOnePercent = hits / numTestInstances * 100
  println("")
  println("precision@1 for all baskets: $precisionAtOnePercent")

  # Compute overall popularity-weighted precision@1
  popWeightedPrecisionAtOnePercent =
    computePopularityWeightedPrecisionAtK(itemPopWeights, resultsForTestInstancesDict,
                                          testInstances, predictedRanks, 1)
  println("")
  println("Popularity-weighted precision@1 for all baskets: $popWeightedPrecisionAtOnePercent")

  # Compute precision@1 (accuracy) for each basket size
  for basketSize in sort(collect(keys(basketSizeResultsDict)))
    basketSizeResult = basketSizeResultsDict[basketSize]
    hits = length(find(x -> x < 1, basketSizeResult.predictedRanks))
    precisionAtOnePercent = hits / basketSizeResult.numBasketInstances * 100
    println("For basket size $basketSize, precision@1: $precisionAtOnePercent")
  end

  # Compute overall precision@5
  hits = length(find(x -> x < 5, predictedRanks))
  precisionAtFivePercent = hits / numTestInstances * 100
  println("")
  println("precision@5 for all baskets: $precisionAtFivePercent")

  # Compute overall popularity-weighted precision@5
  popWeightedPrecisionAtFivePercent =
    computePopularityWeightedPrecisionAtK(itemPopWeights, resultsForTestInstancesDict,
                                          testInstances, predictedRanks, 5)
  println("")
  println("Popularity-weighted precision@5 for all baskets: $popWeightedPrecisionAtFivePercent")

  # Compute precision@5 for each basket size
  for basketSize in sort(collect(keys(basketSizeResultsDict)))
    basketSizeResult = basketSizeResultsDict[basketSize]
    hits = length(find(x -> x < 5, basketSizeResult.predictedRanks))
    precisionAtFivePercent = hits / basketSizeResult.numBasketInstances * 100
    println("For basket size $basketSize, precision@5: $precisionAtFivePercent")
  end

  # Compute overall precision@10
  hits = length(find(x -> x < 10, predictedRanks))
  precisionAtTenPercent = hits / numTestInstances * 100
  println("")
  println("precision@10 for all baskets: $precisionAtTenPercent")

  # Compute overall popularity-weighted precision@10
  popWeightedPrecisionAtTenPercent =
    computePopularityWeightedPrecisionAtK(itemPopWeights, resultsForTestInstancesDict,
                                          testInstances, predictedRanks, 10)
  println("")
  println("Popularity-weighted precision@10 for all baskets: $popWeightedPrecisionAtTenPercent")

  # Compute precision@10 for each basket size
  for basketSize in sort(collect(keys(basketSizeResultsDict)))
    basketSizeResult = basketSizeResultsDict[basketSize]
    hits = length(find(x -> x < 10, basketSizeResult.predictedRanks))
    precisionAtTenPercent = hits / basketSizeResult.numBasketInstances * 100
    println("For basket size $basketSize, precision@10: $precisionAtTenPercent")
  end

  # Compute overall precision@20
  hits = length(find(x -> x < 20, predictedRanks))
  precisionAtTwentyPercent = hits / numTestInstances * 100
  println("")
  println("precision@20 for all baskets: $precisionAtTwentyPercent")

  # Compute overall popularity-weighted precision@20
  popWeightedPrecisionAtTwentyPercent =
    computePopularityWeightedPrecisionAtK(itemPopWeights, resultsForTestInstancesDict,
                                          testInstances, predictedRanks, 20)
  println("")
  println("Popularity-weighted precision@20 for all baskets: $popWeightedPrecisionAtTwentyPercent")

  # Compute precision@20 for each basket size
  for basketSize in sort(collect(keys(basketSizeResultsDict)))
    basketSizeResult = basketSizeResultsDict[basketSize]
    hits = length(find(x -> x < 20, basketSizeResult.predictedRanks))
    precisionAtTwentyPercent = hits / basketSizeResult.numBasketInstances * 100
    println("For basket size $basketSize, precision@20: $precisionAtTwentyPercent")
  end
end

# Computes the popularity weighted (stratified) precision@k
function computePopularityWeightedPrecisionAtK(itemPopWeights, resultsForTestInstancesDict,
                                               testInstances, predictedRanks, k)
  popWeightedHits = 0
  numTestInstances = length(testInstances)
  sumPopWeightsForTestRatings = 0

  startingHeldOutItemId = 0

  for i = 1:numTestInstances
    testInstance = testInstances[i]
    heldOutTestItemId = resultsForTestInstancesDict[testInstance].actualNextItemId - startingHeldOutItemId

    sumPopWeightsForTestRatings += itemPopWeights[heldOutTestItemId]

    if predictedRanks[i] < k
      popWeightedHits += itemPopWeights[heldOutTestItemId]
    end
  end

  popWeightedPrecisionAtKPercent = popWeightedHits / sumPopWeightsForTestRatings * 100
  return popWeightedPrecisionAtKPercent
end

end
