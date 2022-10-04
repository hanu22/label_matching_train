$env:acr = ""
$env:repo = ""
$env:AzureWebJobsAzureWebJobsStorage = ""
$env:TRANSLATOR_API_KEY = "39e1edf34c3648adac64b9e4143b5098"

az acr login --name $env:acr
