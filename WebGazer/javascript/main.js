var dots = [];
var ixFrame = 0;
var videoType = 0;

var secsCount = 0;
var frameCount = 0;

var widthVideo = 320;
var heightVideo = 240;
var durationVideo = 14;

var jsonObj = {
    "method" : "webgazer",
    "task_id": "476",
    "tester_id": "d89041706524f68eab9ee1dfpablox01"
};

var clockReset = -1;

var spaceX = 0;
var spaceY = 0;
var limitX = 0;
var limitY = 0;

var allowFPS = 4;



/*
0 - initial face localization
1 - callibration
2 - content display
*/
window.onload = function() {
    $("#content_video").hide();

    const constraint = {
        audio: true,
        video: {width:widthVideo, height:heightVideo}
    }

    jsonObj.original_media_width = widthVideo;
    jsonObj.original_media_height = heightVideo;

    //Set up the webgazer video feedback.
    var setup = function() {

        //Set up the main canvas. The main canvas is used to calibrate the webgazer.
        var canvas = document.getElementById("plotting_canvas");
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        canvas.style.position = 'fixed';

        jsonObj.calibration_screen_width = canvas.width;
        jsonObj.calibration_screen_height = canvas.height - 50;

        var content_video = document.getElementById("content_video");
        content_video.height = window.innerHeight - 50;
        content_video.width = content_video.height/(heightVideo/widthVideo);
        content_video.style.position = 'fixed';

        jsonObj.displayed_media_width = content_video.width;
        jsonObj.displayed_media_height = content_video.height;

        spaceX = (jsonObj.calibration_screen_width - jsonObj.displayed_media_width)/2;
        spaceY = jsonObj.displayed_media_height - jsonObj.displayed_media_height;

        limitX = spaceX + jsonObj.displayed_media_width;
        limitY = spaceY + jsonObj.displayed_media_height;

        $("#videoDiv").css("left", spaceX + "px");

        jsonObj.relative_initial_media_x = spaceX;
        jsonObj.relative_initial_media_y = spaceY;
        
    };

    function checkIfReady() {
        if (webgazer.isReady()) {
            setup();
        } else {
            setTimeout(checkIfReady, 100);
        }
    }
    setTimeout(checkIfReady,1000);

};

function getDots(data, clock){
    if (videoType==2){

        if(clockReset < 0){
            clockReset = clock;
        }

        
        var xx = 0; 
        var yy = 0;
        //var greenMask = 0;
        //var vv = Math.floor((Math.random() * 10) + 1);
        var sec = Math.round((clock - clockReset + 1)/1000);

        //retrieve information IF ONLY IF the face was detected
        if(data==null){
            xx = -1;
            yy = -1;
        }
        else{
            //greenMask = 1;
            if(data.x > 0)
                xx = data.x;
            if(data.y > 0)
                yy = data.y;    
        }

        //seconds counting
        if(frameCount == 0){
            secsCount = sec;
            frameCount++;
        }
        else{
            if(secsCount < sec)
                frameCount = 0;
            else    
                frameCount++;
        }        

        //normalization
        //var slope = (output_end - output_start) / (input_end - input_start)
        //var output = output_start + slope * (input - input_start)

        //var slope = (output_end/input_end)*input
        
        //storage only the firs 4 frames
        if(frameCount < allowFPS){
            if(xx > spaceX && xx < limitX && yy > spaceY && yy < limitY)
                dots.push({
                    second: sec
                    ,frame: ixFrame
                    ,x: Math.round(((xx - spaceX) / jsonObj.displayed_media_width ) * jsonObj.original_media_width)
                    ,y: Math.round(((yy - spaceY) / jsonObj.displayed_media_height ) * jsonObj.original_media_height)
                });    
            else
                dots.push({
                    second: sec
                    ,frame: ixFrame
                    ,x: -1
                    ,y: -1
                });    
        }

        
        
        ixFrame++;
    }
}

function SaveDots(fileName){

    jsonObj.media_duration = durationVideo;
    jsonObj.data = dots;

    $.ajax({
        type: 'POST',
        url: "saveDots.php",
        data: {
            something: JSON.stringify(jsonObj)
        },
        success: function(result) {
            $("#content_video").hide();
            console.log('the data was successfully sent to the server');
        }
    });

}

window.onbeforeunload = function() {
    //webgazer.end(); //Uncomment if you want to save the data even if you reload the page.
    window.localStorage.clear(); //Comment out if you want to save data across different sessions
}

/**
 * Restart the calibration process by clearing the local storage and reseting the calibration point
 */
function Restart(){
    document.getElementById("Accuracy").innerHTML = "<a>Not yet Calibrated</a>";
    ClearCalibration();
    PopUpInstruction();
}


function PlayPause(){
    var vid = document.getElementById("content_video"); 
    if(vid.paused){
        $("#content_video").show();
        vid.play();
    }
    else
        vid.pause();
}