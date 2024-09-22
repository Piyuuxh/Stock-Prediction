from django import forms

class StockSearchForm(forms.Form):
    ticker = forms.CharField(label='Stock Name', max_length=10)
