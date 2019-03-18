using DelimitedFiles
using Random
using Serialization

export Basket, convertSparseCsvToBaskets, convertBasketsToSparseCsv

struct Basket
  basketItems::Vector{Int}
  heldOutItem::Int
  numItemsInCatalog::Int
end

# Converts data from a file in sparse CSV basket format to the data format
# (sparse vector representation) expected by the DPP learning implementation
function convertSparseCsvToBaskets(csvBasketDataFileName,
  trainingBasketsDictFileName, testBasketsDictFileName, trainingSetSizePercent)
  Random.seed!(1234)

  csvBasketData = readdlm(csvBasketDataFileName, ',')
  numCsvRecords = size(csvBasketData, 1)
  numCsvColumns = size(csvBasketData, 2)
  numItems = 0
  basketItemsDict = Dict{Int, Vector{Int}}()

  # Populate basketItemsDict with baskets from csvBasketData
  basketCounter = 1
  for i = 1:numCsvRecords
    j = 1
    basket = Int[]

    for j = 1:numCsvColumns
      if csvBasketData[i, j] != ""
        itemId = csvBasketData[i, j]
        push!(basket, itemId)

        # Update the number of items in the catalog
        if itemId > numItems
          numItems = itemId
        end
      else
        # We've reached the end of the item entries for this basket
        break
      end
    end

    basketItemsDict[basketCounter] = basket

    basketCounter += 1
  end

  println("num items in catalog: $numItems")

  # Generate histogram of number of baskets for each basket length (item count)
  numBasketsPerBasketLengthHistDict = Dict{Int, Int}()
  for basketId in keys(basketItemsDict)
    itemsInBasket = basketItemsDict[basketId]
    itemCountForBasket = length(itemsInBasket)

    if haskey(numBasketsPerBasketLengthHistDict, itemCountForBasket)
      # There is already at least one basket with this item count
      numBasketsWithItemCount = numBasketsPerBasketLengthHistDict[itemCountForBasket]
      numBasketsWithItemCount += 1
      numBasketsPerBasketLengthHistDict[itemCountForBasket] = numBasketsWithItemCount
    else
      numBasketsPerBasketLengthHistDict[itemCountForBasket] = 1
    end
  end

  for itemCountForBasket in sort(collect(keys(numBasketsPerBasketLengthHistDict)))
    println("Item count $itemCountForBasket: $(numBasketsPerBasketLengthHistDict[itemCountForBasket])")
  end

  # Remove baskets of size 1 from basketItemsDict, since we are not interested in them for a DPP
  for basketId in keys(basketItemsDict)
    itemsInBasket = basketItemsDict[basketId]
    if length(itemsInBasket) == 1
      delete!(basketItemsDict, basketId)
    end
  end

  # Generate training and test sets
  trainingBasketsDict = Dict{Int, Basket}()
  testBasketsDict = Dict{Int, Basket}()
  numBaskets = length(collect(keys(basketItemsDict)))
  trainingSetNumBaskets = convert(Int, round(numBaskets * trainingSetSizePercent));
  testSetNumBaskets = numBaskets - trainingSetNumBaskets
  shuffledBasketItemsDictKeys = shuffle(collect(keys(basketItemsDict)))
  trainingSetBaskets = shuffledBasketItemsDictKeys[1:trainingSetNumBaskets]
  testSetBaskets = shuffledBasketItemsDictKeys[(trainingSetNumBaskets + 1):numBaskets]
  for basketId in trainingSetBaskets
    basketItems = shuffle(basketItemsDict[basketId])
    heldOutItem = pop!(basketItems)
    basket = Basket(basketItems, heldOutItem, numItems)
    trainingBasketsDict[basketId] = basket
  end
  for basketId in testSetBaskets
    basketItems = shuffle(basketItemsDict[basketId])
    heldOutItem = pop!(basketItems)
    basket = Basket(basketItems, heldOutItem, numItems)
    testBasketsDict[basketId] = basket
  end

  println("Num baskets in training set: $(length(collect(keys(trainingBasketsDict))))")
  println("Num baskets in test set: $(length(collect(keys(testBasketsDict))))")

  open(f -> serialize(f, trainingBasketsDict), trainingBasketsDictFileName, "w");
  println("Saved trainingBasketsDict")

  open(f -> serialize(f, testBasketsDict), testBasketsDictFileName, "w");
  println("Saved testBasketsDict")
end

# Converts data from a file in sparse vector (DPP implementation) format to
# sparse CSV format
function convertBasketsToSparseCsv(basketsDictFileName, basketsDictObjectName,
                                   csvBasketDataFileName)
  # Load basket data
  basketsDict = load(basketsDictFileName, basketsDictObjectName)
  println("Loaded $basketsDictFileName")

  # Write basket data to csvBasketDataFileName
  csvBasketDataFile = open(csvBasketDataFileName, "w")
  for basketId in collect(keys(basketsDict))
    basket = basketsDict[basketId]
    basketArray = basket.basketItems
    push!(basketArray, basket.heldOutItem)

    println("Contents for basket $basketId: $basketArray")
    write(csvBasketDataFile, join(sort(basketArray), " "), "\n")
  end
  close(csvBasketDataFile)
end
