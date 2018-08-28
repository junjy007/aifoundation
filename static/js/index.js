function handleError(error) {
    console.error('navigator.getUserMedia error: ', error);
}

const constraints = {
    video: true
};
(function () {
    const captureVideoButton =
        document.querySelector('#screenshot .capture-button');
    const screenshotButton = document.querySelector('#screenshot-button');
    img = document.querySelector('#screenshot img');
    const video = document.querySelector('#screenshot video');

    const canvas = document.createElement('canvas');

    captureVideoButton.onclick = function () {
        navigator.mediaDevices.getUserMedia(constraints).then(handleSuccess).catch(handleError);
    };

    screenshotButton.onclick = video.onclick = function () {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        // Other browsers will fall back to image/png
        img.src = canvas.toDataURL('image/png');
    };

    function handleSuccess(stream) {
        screenshotButton.disabled = false;
        video.srcObject = stream;
    }
})();

function upload() {
    /*
     var video = document.querySelector('#screenshot video');
     var scratchCanvas = document.createElement('canvas');
     scratchCanvas.width = video.videoWidth;
     scratchCanvas.height = video.videoHeight;
     var context = scratchCanvas.getContext('2d');
     context.drawImage(video, 0, 0);
     var dataURL = scratchCanvas.toDataURL('image/png');
     // console.log(dataURL.length)*/
    const img = document.querySelector('#screenshot img');
    var processMethod=$('#model option:selected').val();
    $.ajax({
        type: 'POST',
        url: '/upload',
        data: {imageBase64: img.src, processMethod: processMethod},
        success: function (msg) {
            $('#four').html('<img src='+msg+' width="400px" height="300px"/>');

        }
    });
}

var ____configArray;

function __initDropDownList(configArray) {
    //获取Select菜单
    ____configArray = configArray;
    var existArray = configArray.split("|");
    for (var i = 0; i < existArray.length; i++) {
        if (existArray[i].length < 1) {
            return;
        }
        //根据参数分别获取div，并分别添加事件
        var parentContainer = document.getElementById(existArray[i]);
        if (!parentContainer) {
            return;
        }
        //获取下面的select，且获取其中的option
        var selectObj = parentContainer.getElementsByTagName("select");
        if (selectObj.length < 1) {
            return;
        }
        var optionArray = selectObj[0].getElementsByTagName("option");
        //获取option，并分别添加到各个li
        var optionLength = optionArray.length;
        for (var j = 0; j < optionLength; j++) {
            //获取ul，以便能够添加项目
            var ulObj = parentContainer.getElementsByTagName("ul");
            if (ulObj.length < 1) {
                return;
            }
            //获取span，以便能显示当前选择的项目
            var spanObj = parentContainer.getElementsByTagName("span");
            if (spanObj.length < 1) {
                return;
            }
            var liObj = document.createElement("li");
            var textNode = document.createTextNode(optionArray[j].firstChild.nodeValue)
            liObj.appendChild(textNode);
            liObj.setAttribute("currentIndex", j);
            //如果option的selected="selected"
            if (optionArray[j].selected) {
                selectCurrentItem(ulObj[0], liObj);
            }
            //给每个liObj添加事件
            liObj.onclick = function () {
                selectCurrentItem(this.parentNode, this);
            }
            liObj.onmouseover = function () {
                if (this.className.indexOf("current") < 0) {
                    this.className = "over";
                }
            }
            liObj.onmouseout = function () {
                if (this.className.indexOf("current") < 0) {
                    this.className = "normal";
                }
            }
            ulObj[0].appendChild(liObj);
            spanObj[0].onclick = function (event) {
                //如果当前是显示的，就隐藏，反之亦然
                showHiddenUl(this);
            }
            spanObj[0].onmouseover = function () {
                this.className = 'over';
            }
            spanObj[0].onmouseout = function () {
                this.className = "";
            };
            ulObj[0].onclick = function () {
                this.className = "";
            }
        }
        parentContainer.onclick = function (event) {
            if (!event) {
                event = window.event;
            }
            event.cancelBubble = true;
            var eventUlObj = this.getElementsByTagName("ul")[0];
            bodyClickHiddenUl(eventUlObj);
        }
    }
}

function selectCurrentItem(ulObj, currentObj) {
    var parentObj = ulObj.parentNode;
    var spanObj = parentObj.getElementsByTagName("span")[0];
    spanObj.firstChild.nodeValue = currentObj.firstChild.nodeValue;
    var selectObj = parentObj.getElementsByTagName("select")[0];
    selectObj.selectedIndex = parseInt(currentObj.getAttribute("currentIndex"));
    var ulLiObj = ulObj.getElementsByTagName("li");
    var length = ulLiObj.length;
    var currentLiObj = null;
    for (var i = 0; i < length; i++) {
        currentLiObj = ulLiObj[i];
        currentLiObj.className = "normal";
    }
    currentObj.className = "current";

    var t=currentObj.firstChild.nodeValue;
    if(t=='MNIST'){
    	$('#two')[0].innerHTML='\
						<p>模型名称:cnn</p>\
						<p>描述：手写数字识别</p>\
						<p>输入： 单通道，28*28像素</p>\
						<p>输出：数字</p>';
    }
    else if(t=='openpose'){
    	$('#two')[0].innerHTML='\
						<p>模型名称：openpose</p>\
						<p>描述：人的骨架识别</p>\
						<p>输入： 任意大小</p>\
						<p>输出：骨架图</p>';
    }else if(t=='UNIT(city2gta)'){
    	$('#two')[0].innerHTML='\
						<p>模型名称：UNIT</p>\
						<p>描述：现实图像到虚拟图像的转换</p>\
						<p>输入： 任意大小，现实城市场景</p>\
						<p>输出：虚拟城市场景</p>';
    }else if(t=='UNIT(gta2city)'){
    	$('#two')[0].innerHTML='\
						<p>模型名称：UNIT</p>\
						<p>描述：虚拟图像到现实图像的转换</p>\
						<p>输入： 任意大小，虚拟城市场景</p>\
						<p>输出：现实城市场景</p>';
    }else if(t=='yolo3'){
    	$('#two')[0].innerHTML='\
						<p>模型名称：yolo3</p>\
						<p>描述：目标检测</p>\
						<p>输入： 任意大小</p>\
						<p>输出：目标区域被标识的图像</p>';
    }else if(t=='ToStyle(night)'){
    	$('#two')[0].innerHTML='\
						<p>描述：图像风格转换</p>\
						<p>输入： 任意大小</p>\
						<p>输出：starry-night风格图像</p>';
    }else if(t=='ToStyle(mosaic)'){
    	$('#two')[0].innerHTML='\
						<p>描述：图像风格转换</p>\
						<p>输入： 任意大小</p>\
						<p>输出：mosaic风格图像</p>';
    }else if(t=='ToStyle(udnie)'){
    	$('#two')[0].innerHTML='\
						<p>描述：图像风格转换</p>\
						<p>输入： 任意大小</p>\
						<p>输出：udnie风格图像</p>';
    }else if(t=='ToStyle(candy)'){
    	$('#two')[0].innerHTML='\
						<p>描述：图像风格转换</p>\
						<p>输入： 任意大小</p>\
						<p>输出：candy风格图像</p>';
    }
}

function showHiddenUl(currentObj) {
    var parentNode = currentObj.parentNode;
    var ulObj = parentNode.getElementsByTagName("ul")[0];
    if (ulObj.className == "") {
        ulObj.className = "show";
    } else {
        ulObj.className = "";
    }
}

//点击body区域（非“下拉菜单”）隐藏菜单
function addBodyClick(func) {
    var bodyObj = document.getElementsByTagName("body")[0];
    var oldBodyClick = bodyObj.onclick;
    if (typeof bodyObj.onclick != 'function') {
        bodyObj.onclick = func;
    } else {
        bodyObj.onclick = function () {
            oldBodyClick();
            func();
        }
    }
}

//隐藏所有的UL
function bodyClickHiddenUl(eventUlObj) {
    var existArray = ____configArray.split("|");
    for (var i = 0; i < existArray.length; i++) {
        if (existArray[i].length < 1) {
            return;
        }
        //寻找所有UL并且隐藏
        var parentContainer = document.getElementById(existArray[i]);
        if (!parentContainer) {
            return;
        }
        var ulObj = parentContainer.getElementsByTagName("ul");
        if (eventUlObj != ulObj[0]) {
            ulObj[0].className = "";
        }
    }
}

var __dropDownList = "dropDownList1";
__initDropDownList(__dropDownList);
//添加这个可以确保点击body区域的时候 也可以隐藏菜单
addBodyClick(bodyClickHiddenUl);


var dropBox;
  window.onload=function(){
   dropBox = document.getElementById("three");
   // 鼠标进入放置区时
   dropBox.ondragenter = ignoreDrag;
   // 拖动文件的鼠标指针位置放置区之上时发生
   dropBox.ondragover = ignoreDrag;
   dropBox.ondrop = drop;
  }
  function ignoreDrag(e){
   // 确保其他元素不会取得该事件
   e.stopPropagation();
   e.preventDefault();
  }
  function drop(e){
   e.stopPropagation();
   e.preventDefault();

   // 取得拖放进来的文件
   var data = e.dataTransfer;
   var files = data.files;
   // 将其传给真正的处理文件的函数

   var file = files[0];
   var reader = new FileReader();
   reader.onload=function(e){
       $('#screenshot-img').attr('src',e.target.result);
   }
   reader.readAsDataURL(file);
  }