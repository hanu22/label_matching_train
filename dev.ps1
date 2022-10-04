$env:acr = ""
$env:repo = ""
$env:AzureWebJobsAzureWebJobsStorage = ""
$env:TRANSLATOR_API_KEY = "3b2c862123f1485282f0f0a761178a1c"

az acr login --name $env:acr
