from django.shortcuts import render
from .forms import StockSearchForm
from .stock_predictor import analyze_stock  # Assuming stock_predictor.py contains your previous logic
from django.contrib import messages

def home(request):
    if request.method == 'POST':
        form = StockSearchForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            try:
                context = analyze_stock(ticker, "2023-01-01", "2024-09-18")
                return render(request, 'predictor/results.html', context)
            except ValueError as e:
                messages.error(request, str(e))  # Display the error message
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = StockSearchForm()
    return render(request, 'predictor/home.html', {'form': form})

