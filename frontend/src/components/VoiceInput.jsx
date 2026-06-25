import { useState, useRef, useCallback } from 'react'
import './VoiceInput.css'

const API_BASE = '/api'

function VoiceInput({ onResult }) {
    const [isRecording, setIsRecording] = useState(false)
    const [isProcessing, setIsProcessing] = useState(false)
    const mediaRecorderRef = useRef(null)
    const audioChunksRef = useRef([])

    const startRecording = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
            mediaRecorderRef.current = new MediaRecorder(stream)
            audioChunksRef.current = []

            mediaRecorderRef.current.ondataavailable = (event) => {
                audioChunksRef.current.push(event.data)
            }

            mediaRecorderRef.current.onstop = async () => {
                setIsProcessing(true)
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })

                try {
                    const formData = new FormData()
                    formData.append('audio', audioBlob, 'recording.wav')

                    const response = await fetch(`${API_BASE}/transcribe`, {
                        method: 'POST',
                        body: formData,
                    })

                    if (response.ok) {
                        const data = await response.json()
                        if (data.text) {
                            onResult(data.text)
                        }
                    }
                } catch (err) {
                    console.error('Transcription error:', err)
                } finally {
                    setIsProcessing(false)
                }

                // Stop all tracks
                stream.getTracks().forEach(track => track.stop())
            }

            mediaRecorderRef.current.start()
            setIsRecording(true)
        } catch (err) {
            console.error('Microphone error:', err)
            alert('Unable to access microphone. Please check permissions.')
        }
    }, [onResult])

    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop()
            setIsRecording(false)
        }
    }, [isRecording])

    const handleClick = () => {
        if (isRecording) {
            stopRecording()
        } else {
            startRecording()
        }
    }

    return (
        <div className="voice-input">
            <button
                className={`voice-button ${isRecording ? 'recording' : ''} ${isProcessing ? 'processing' : ''}`}
                onClick={handleClick}
                disabled={isProcessing}
            >
                {isProcessing ? (
                    <>
                        <span className="voice-icon">⏳</span>
                        <span>Processing...</span>
                    </>
                ) : isRecording ? (
                    <>
                        <span className="voice-icon pulse">🎙️</span>
                        <span>Stop Recording</span>
                    </>
                ) : (
                    <>
                        <span className="voice-icon">🎤</span>
                        <span>Voice Search</span>
                    </>
                )}
            </button>
            {isRecording && (
                <div className="recording-indicator">
                    <span className="dot"></span>
                    <span>Listening...</span>
                </div>
            )}
        </div>
    )
}

export default VoiceInput
