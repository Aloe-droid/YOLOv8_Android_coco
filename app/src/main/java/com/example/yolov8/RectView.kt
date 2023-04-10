package com.example.yolov8

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.round

class RectView(context: Context, attributeSet: AttributeSet) : View(context, attributeSet) {

    private var results: ArrayList<Result>? = null
    private lateinit var classes: Array<String>

    private val textPaint = Paint().also {
        it.textSize = 60f
        it.color = Color.WHITE
    }

    fun transformRect(results: ArrayList<Result>) {
        // scale 구하기
        val scaleX = width / DataProcess.INPUT_SIZE.toFloat()
        val scaleY = scaleX * 9f / 16f
        val realY = width * 9f / 16f
        val diffY = realY - height

        results.forEach {
            it.rectF.left *= scaleX
            it.rectF.right *= scaleX
            it.rectF.top = it.rectF.top * scaleY - (diffY / 2f)
            it.rectF.bottom = it.rectF.bottom * scaleY - (diffY / 2f)
        }
        this.results = results
    }

    override fun onDraw(canvas: Canvas?) {
        //그림 그리기
        results?.forEach {
            canvas?.drawRect(it.rectF, findPaint(it.classIndex))
            canvas?.drawText(
                classes[it.classIndex] + ", " + round(it.score * 100) + "%",
                it.rectF.left + 10,
                it.rectF.top + 60,
                textPaint
            )
        }
        super.onDraw(canvas)
    }

    fun setClassLabel(classes: Array<String>) {
        this.classes = classes
    }

    //paint 지정
    private fun findPaint(classIndex: Int): Paint {
        val paint = Paint()
        paint.style = Paint.Style.STROKE    // 빈 사각형 그림
        paint.strokeWidth = 10.0f           // 굵기 10
        paint.strokeCap = Paint.Cap.ROUND   // 끝을 뭉특하게
        paint.strokeJoin = Paint.Join.ROUND // 끝 주위도 뭉특하게
        paint.strokeMiter = 100f            // 뭉특한 정도는 100도

        //임의로 지정한 색상
        paint.color = when (classIndex) {
            0, 45, 18, 19, 22, 30, 42, 43, 44, 61, 71, 72 -> Color.WHITE
            1, 3, 14, 25, 37, 38, 79 -> Color.BLUE
            2, 9, 10, 11, 32, 47, 49, 51, 52 -> Color.RED
            5, 23, 46, 48 -> Color.YELLOW
            6, 13, 34, 35, 36, 54, 59, 60, 73, 77, 78 -> Color.GRAY
            7, 24, 26, 27, 28, 62, 64, 65, 66, 67, 68, 69, 74, 75 -> Color.BLACK
            12, 29, 33, 39, 41, 58, 50 -> Color.GREEN
            15, 16, 17, 20, 21, 31, 40, 55, 57, 63 -> Color.DKGRAY
            70, 76 -> Color.LTGRAY
            else -> Color.DKGRAY
        }
        return paint
    }
}