import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
  Animated,
  Easing,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import { Audio } from 'expo-av';
import { useLanguage } from '@/components/LanguageContext';
import translations from '@/components/translation';

const AIChat = () => {
  const [messages, setMessages] = useState([
    {
      id: '1',
      text: "Hello! I'm your AushadX medical assistant. How can I help you with your medical questions today?",
      sender: 'bot',
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [recording, setRecording] = useState(null);
  const { language } = useLanguage();
  const navigation = useNavigation();
  const flatListRef = useRef(null);

  const t = (key) => translations[language]?.[key] || key;

  useEffect(() => {
    if (flatListRef.current && messages.length > 0) {
      setTimeout(() => {
        flatListRef.current.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

  // Start Recording
  const startRecording = async () => {
    try {
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        alert('Permission to access microphone is required!');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const recordingObject = new Audio.Recording();
      await recordingObject.prepareToRecordAsync(Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY);
      await recordingObject.startAsync();

      setRecording(recordingObject);
    } catch (err) {
      console.error('Failed to start recording', err);
    }
  };

  // Stop Recording and Send to API
  const stopRecording = async () => {
    if (!recording) return;

    setIsProcessing(true);
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    setRecording(null);

    const formData = new FormData();
    formData.append('file', {
      uri,
      type: 'audio/wav',
      name: 'audio.wav',
    });

    try {
      const response = await fetch('http://192.168.108.195:8000/process-audio', {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        body: formData,
      });

      const data = await response.json();
      const botResponse = {
        id: (Date.now() + 1).toString(),
        text: data.response || "Couldn't process your voice input.",
        sender: 'bot',
      };

      setMessages((prevMessages) => [...prevMessages, botResponse]);
    } catch (error) {
      console.error('Error sending audio:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { id: (Date.now() + 1).toString(), text: 'Error processing audio.', sender: 'bot' },
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <LinearGradient colors={['#121212', '#1E1E1E', '#252525']} style={styles.gradientBackground}>
        <View style={styles.header}>
          <TouchableOpacity onPress={navigation.goBack} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{t('AushadX Assistant')}</Text>
          <TouchableOpacity onPress={() => setMessages([])} style={styles.clearButton}>
            <Ionicons name="trash-outline" size={22} color="white" />
          </TouchableOpacity>
        </View>

        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <View style={[styles.messageBubble, item.sender === 'user' ? styles.userMessage : styles.botMessage]}>
              <Text style={styles.messageText}>{item.text}</Text>
            </View>
          )}
          contentContainerStyle={styles.messageList}
          showsVerticalScrollIndicator={false}
        />

        <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.inputContainer}>
          <View style={styles.inputRow}>
            <TouchableOpacity onPress={recording ? stopRecording : startRecording} style={styles.voiceButton}>
              <Ionicons name={recording ? 'stop-circle' : 'mic'} size={28} color="white" />
            </TouchableOpacity>
            <TextInput
              style={styles.input}
              value={inputText}
              onChangeText={setInputText}
              placeholder={t('Type your medical question...')}
              placeholderTextColor="#9e9e9e"
              returnKeyType="send"
            />
            <TouchableOpacity style={styles.sendButton} onPress={() => {}} activeOpacity={0.7}>
              <Ionicons name="send" size={24} color="white" />
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </LinearGradient>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#121212' },
  gradientBackground: { flex: 1, width: '100%' },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 16 },
  headerTitle: { fontSize: 18, fontWeight: '600', color: 'white' },
  backButton: { padding: 8 },
  clearButton: { padding: 8 },
  messageList: { paddingBottom: 20 },
  messageBubble: { maxWidth: '80%', padding: 12, borderRadius: 16, marginBottom: 10 },
  userMessage: { alignSelf: 'flex-end', backgroundColor: '#4466EE' },
  botMessage: { alignSelf: 'flex-start', backgroundColor: '#2C2C2E' },
  messageText: { color: 'white', fontSize: 16 },
  inputContainer: { width: '100%', marginTop: 'auto', padding: 10 },
  inputRow: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#2C2C2E', borderRadius: 30, padding: 10 },
  input: { flex: 1, color: 'white', fontSize: 16, marginLeft: 10 },
  sendButton: { padding: 8 },
  voiceButton: { padding: 8, backgroundColor: '#4466EE', borderRadius: 20, marginRight: 8 },
});

export default AIChat;
