/**
 * Created by Eric Lind on 3/1/16.
 */



// Stop form submission from refreshing page
$('#classify-form').on('submit', function(e) {
    e.preventDefault();

    var txt = $(this).find('#classify-txt').val();

    if (txt) {
        console.log("Sending text");
        // Send the search
        $.ajax({
            method: 'POST',
            url:    '/classify',
            contentType: 'text/plain',
            data: txt,
            success: processClassification,
            error: handleError
        });
    } else {
        $('#classification').text('Enter some text!');
    }


});

/**
 * Callback for when a search for emails throws an error
 * @param jqXHR
 * @param textStatus
 * @param errorThrown
 * @private
 */
function handleError(jqXHR, textStatus, errorThrown) {
    console.log("Error getting classification from server!", textStatus, errorThrown);
    alert("Error getting data from server!");
}

function processClassification(data) {
    $('#classification').text(data)
}
