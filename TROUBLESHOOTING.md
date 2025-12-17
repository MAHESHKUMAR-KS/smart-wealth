# Troubleshooting: WealthyWise Assistant Not Appearing in UI

If the WealthyWise Assistant chatbot is not appearing in your application UI, here are the steps to diagnose and fix the issue:

## 1. Verify Module Imports

Check that all required modules are properly imported in `app_professional.py`:

```python
# Look for these import statements near the top of the file
from ai_risk_persona import get_investor_persona
from wealthywise_assistant import render_chat_interface
```

## 2. Check Navigation Menu

Verify that "🎓 WealthyWise Assistant" is included in the navigation options:

```python
pages = [
    "🏠 Home Dashboard",
    "📊 Risk Profiler",
    "🎯 Portfolio Recommendations",
    "💹 SIP & Goal Calculator",
    "🔍 Fund Explorer",
    "🎓 WealthyWise Assistant",  # Make sure this line exists
    "📚 Learning Center"
]
```

## 3. Confirm Page Routing

Ensure the routing logic includes the assistant page:

```python
elif page == "🎓 WealthyWise Assistant":
    render_chat_interface()
```

## 4. Test Module Loading

Run this command to verify the assistant module loads correctly:

```bash
python -c "from wealthywise_assistant import render_chat_interface; print('SUCCESS: Module loaded')"
```

## 5. Restart the Application

After making any changes, completely restart the Streamlit application:

```bash
streamlit run app_professional.py
```

## 6. Clear Browser Cache

Sometimes browser caching can cause UI elements not to appear:
- Hard refresh your browser (Ctrl+F5 or Cmd+Shift+R)
- Or open the app in an incognito/private browsing window

## 7. Check for Errors in Console

Look at the terminal/console where you started the Streamlit app for any error messages:
- ImportError messages
- Syntax errors
- Runtime exceptions

## 8. Verify File Locations

Ensure all files are in the correct locations:
- `wealthywise_assistant.py` should be in the same directory as `app_professional.py`
- Both files should be in the root of your project

## 9. Test the Assistant Directly

You can test the assistant interface directly by running:

```bash
streamlit run wealthywise_assistant.py
```

This will show just the chatbot interface and confirm it works independently.

## 10. Common Issues and Solutions

### Issue: Import Error
**Symptom**: "ModuleNotFoundError" when starting the app
**Solution**: Ensure `wealthywise_assistant.py` is in the same directory as `app_professional.py`

### Issue: Navigation Not Showing
**Symptom**: The option doesn't appear in the dropdown
**Solution**: Check that the page is added to the `pages` array and that the `format_func` is working correctly

### Issue: Blank Page When Selected
**Symptom**: Clicking the option shows a blank page
**Solution**: Check the routing logic matches exactly with the selectbox values

If you've tried all these steps and still have issues, please share:
1. Any error messages from the console
2. Screenshots of your navigation menu
3. The exact behavior you're seeing