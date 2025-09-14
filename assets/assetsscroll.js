// assets/scroll.js
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        scrollToBottom: function() {
            setTimeout(function() {
                window.scrollTo(0,document.body.scrollHeight);
            }, 500);  // Delay in milliseconds
            return '';
        }
    }
});
