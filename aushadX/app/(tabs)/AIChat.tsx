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
  ActivityIndicator,
  Alert,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import { Audio } from 'expo-av';
import * as Speech from 'expo-speech';
import { useLanguage } from '@/components/LanguageContext';
import translations from '@/components/translation';

// Replace with your Gemini API key
const GEMINI_API_KEY = 'AIzaSyB91pjFVJq3NbLfU0sV4dQPnkl3DQSCqRo';
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent';

const AIChat = () => {
  const [messages, setMessages] = useState([
    {
      id: '1',
      text: "Hello! I'm your AushadX medical assistant. How can I help you with your medical questions today?",
      sender: 'bot',
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [recording, setRecording] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
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

  // Function to send message to Gemini API
  const sendMessageToGemini = async (userMessage) => {
    try {
      setIsLoading(true);
      
      // Simulate API call with medical domain knowledge
      // In a real implementation, you would call the Gemini API here
      await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate API latency
      
      let response;
      
      // Check if the message is related to medical/health domains
      const medicalKeywords = [
        'health', 'doctor', 'medicine', 'symptom', 'pain', 'disease', 'treatment',
        'hospital', 'clinic', 'diagnose', 'drug', 'prescription', 'medical', 'cure',
        'therapy', 'illness', 'condition', 'body', 'organ', 'heart', 'brain', 'cancer',
        'diabetes', 'blood', 'pressure', 'vaccine', 'allergy', 'fever', 'cough', 'cold',
        'flu', 'virus', 'infection', 'surgery', 'wound', 'injury', 'diet', 'nutrition',
        'exercise', 'wellness', 'mental health', 'anxiety', 'depression', 'stress',
        'headache', 'medication', 'pharmacy', 'dosage', 'side effect', 'specialist',
        'vaccination', 'immunity', 'healthcare', 'physician', 'nurse', 'patient'
      ];
      
      const lowerCaseMessage = userMessage.toLowerCase();
      const isMedicalQuery = medicalKeywords.some(keyword => 
        lowerCaseMessage.includes(keyword.toLowerCase())
      );
      
      if (isMedicalQuery) {
        // Respond with relevant medical information
        if (lowerCaseMessage.includes('headache')) {
          response = "Headaches can be caused by various factors including stress, dehydration, lack of sleep, or eye strain. For occasional headaches, rest, hydration, and over-the-counter pain relievers may help. If you experience severe, frequent, or unusual headaches, please consult a healthcare professional.";
        } else if (lowerCaseMessage.includes('cold') || lowerCaseMessage.includes('flu') || lowerCaseMessage.includes('fever')) {
          response = "Common cold and flu symptoms can be managed with rest, hydration, and over-the-counter medications to reduce fever and alleviate symptoms. If symptoms persist for more than a week or worsen suddenly, please consult a healthcare professional.";
        } else if (lowerCaseMessage.includes('blood pressure')) {
          response = "Maintaining healthy blood pressure is important for heart health. Normal blood pressure is typically around 120/80 mmHg. Lifestyle factors like diet, exercise, and stress management can help control blood pressure. If you're concerned about your blood pressure, please consult a healthcare professional for proper monitoring and treatment if necessary.";
        } else if (lowerCaseMessage.includes('diabetes')) {
          response = "Diabetes is a condition that affects how your body uses blood sugar (glucose). Management typically involves monitoring blood sugar levels, medication or insulin therapy, healthy eating, and regular physical activity. Regular check-ups with healthcare providers are essential for managing diabetes effectively.";
        } else if (lowerCaseMessage.includes('covid') || lowerCaseMessage.includes('coronavirus')) {
          response = "COVID-19 symptoms may include fever, cough, fatigue, and loss of taste or smell. If you experience symptoms, consider getting tested and follow local health guidelines. Vaccination remains one of the most effective ways to prevent severe illness from COVID-19.";
        } else if (lowerCaseMessage.includes('mental health') || lowerCaseMessage.includes('anxiety') || lowerCaseMessage.includes('depression')) {
          response = "Mental health is just as important as physical health. Common strategies for managing mental health include therapy, mindfulness practices, adequate sleep, regular exercise, and maintaining social connections. If you're struggling with mental health issues, please reach out to a healthcare professional for proper evaluation and support.";
        } else if (lowerCaseMessage.includes('exercise') || lowerCaseMessage.includes('fitness')) {
          response = "Regular physical activity is essential for good health. Adults should aim for at least 150 minutes of moderate-intensity exercise per week. Before starting any new exercise routine, especially if you have pre-existing health conditions, it's advisable to consult with a healthcare provider.";
        } else if (lowerCaseMessage.includes('diet') || lowerCaseMessage.includes('nutrition')) {
          response = "A balanced diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats supports overall health. Individualized dietary needs may vary based on factors like age, gender, activity level, and health conditions. For personalized nutrition advice, consider consulting with a registered dietitian.";
        }else if (lowerCaseMessage.includes('metabolism')) {
          response = "To boost metabolism naturally, consider increasing physical activity, building muscle through strength training, staying hydrated, getting enough sleep, and eating small, frequent meals that include protein.";
        } else if (lowerCaseMessage.includes('thyroid')) {
          response = "Thyroid issues can affect energy levels, weight, and mood. Symptoms may include fatigue, weight changes, and temperature sensitivity. If you suspect a thyroid problem, consult a healthcare provider for a blood test and diagnosis.";
        } else if (lowerCaseMessage.includes('energy')) {
          response = "Natural ways to boost energy include regular physical activity, proper sleep, a balanced diet, and staying hydrated. Avoid excessive caffeine and focus on managing stress.";
        } else if (lowerCaseMessage.includes('belly fat')) {
          response = "Losing belly fat involves a combination of a healthy diet, regular cardio and strength exercises, stress reduction, and adequate sleep. There’s no quick fix, but consistent lifestyle changes help.";
        } else if (lowerCaseMessage.includes('weight')) {
          response = "Maintaining a healthy weight requires a balance between calorie intake and physical activity. Eating a nutritious diet and staying active are key. Consult with a healthcare professional for personalized advice.";
        } else if (lowerCaseMessage.includes('sugar')) {
          response = "Too much sugar can lead to weight gain, diabetes, and heart disease. Reducing intake of sugary drinks and processed foods can significantly improve overall health.";
        } else if (lowerCaseMessage.includes('inflammation')) {
          response = "To reduce inflammation, focus on an anti-inflammatory diet rich in fruits, vegetables, whole grains, and omega-3 fats. Avoid processed foods and manage stress effectively.";
        } else if (lowerCaseMessage.includes('yoga')) {
          response = "Yoga offers many health benefits, including improved flexibility, stress reduction, better breathing, and mental clarity. It can also aid in managing chronic pain and improving sleep.";
        } else if (lowerCaseMessage.includes('depression')) {
          response = "Symptoms of depression include persistent sadness, loss of interest in activities, fatigue, and changes in appetite or sleep. If you or someone you know is struggling, reach out to a mental health professional.";
        } else if (lowerCaseMessage.includes('bloating')) {
          response = "Bloating may be caused by overeating, gas, or food intolerances. Staying hydrated, eating slowly, and avoiding trigger foods can help relieve bloating. If it persists, consult a doctor.";
        }else if (lowerCaseMessage.includes('food allergies')) {
          response = "Food allergies can cause symptoms like hives, swelling, digestive issues, or even severe reactions. If you suspect a food allergy, see an allergist for testing and advice on managing your diet.";
        } else if (lowerCaseMessage.includes('build muscle')) {
          response = "Building muscle naturally involves consistent strength training, adequate protein intake, proper rest, and balanced nutrition. Avoid overtraining and consider consulting a fitness professional.";
        } else if (lowerCaseMessage.includes('floss') || lowerCaseMessage.includes('brush teeth')) {
          response = "Good dental hygiene involves brushing twice daily, flossing once daily, and regular dental checkups. This helps prevent cavities, gum disease, and bad breath.";
        } else if (lowerCaseMessage.includes('lemon water')) {
          response = "Drinking lemon water may support hydration and provide a small amount of vitamin C. However, it should complement, not replace, a balanced diet.";
        } else if (lowerCaseMessage.includes('cholesterol')) {
          response = "High cholesterol can increase heart disease risk. A healthy diet, regular exercise, and sometimes medication can help manage cholesterol levels. Consult your doctor for testing and guidance.";
        } else if (lowerCaseMessage.includes('detox')) {
          response = "The body naturally detoxifies through the liver and kidneys. Supporting these organs with hydration, a balanced diet, and avoiding toxins is the safest approach. Be cautious with commercial detox products.";
        } else if (lowerCaseMessage.includes('fiber')) {
          response = "Dietary fiber is important for digestion, blood sugar control, and heart health. Include whole grains, fruits, vegetables, and legumes in your diet for sufficient fiber.";
        } else if (lowerCaseMessage.includes('high blood pressure') || lowerCaseMessage.includes('hypertension')) {
          response = "High blood pressure can be managed with lifestyle changes like reducing salt intake, exercising, managing stress, and following medical advice if needed.";
        } else if (lowerCaseMessage.includes('sleep') && lowerCaseMessage.includes('weight gain')) {
          response = "Lack of sleep can disrupt hormones related to hunger and metabolism, potentially leading to weight gain. Prioritizing good sleep hygiene supports overall health.";
        } else if (lowerCaseMessage.includes('green tea')) {
          response = "Green tea contains antioxidants that may support metabolism and heart health. Drinking it in moderation can be part of a healthy lifestyle.";
        }else if (lowerCaseMessage.includes('muscle cramps')) {
          response = "Muscle cramps are often caused by dehydration, electrolyte imbalances, or overuse. Staying hydrated, stretching, and ensuring adequate intake of minerals like potassium and magnesium can help prevent cramps.";
        } else if (lowerCaseMessage.includes('eye strain')) {
          response = "To reduce eye strain, follow the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds. Adjust screen brightness and ensure proper lighting.";
        } else if (lowerCaseMessage.includes('arthritis')) {
          response = "Arthritis symptoms include joint pain and stiffness. Managing arthritis often involves physical therapy, medication, and lifestyle changes like regular low-impact exercise.";
        } else if (lowerCaseMessage.includes('dehydration')) {
          response = "Dehydration can cause headaches, dizziness, and fatigue. Drinking adequate fluids, especially water, throughout the day is essential to stay hydrated.";
        } else if (lowerCaseMessage.includes('heart health')) {
          response = "Maintaining heart health involves a balanced diet, regular exercise, managing stress, avoiding smoking, and regular medical checkups.";
        } else if (lowerCaseMessage.includes('gut health')) {
          response = "Good gut health is supported by a fiber-rich diet, probiotics, hydration, and avoiding excessive processed foods and antibiotics unless necessary.";
        } else if (lowerCaseMessage.includes('stress management')) {
          response = "Effective stress management techniques include mindfulness, exercise, adequate sleep, social support, and time management.";
        } else if (lowerCaseMessage.includes('immune system')) {
          response = "Boost your immune system by eating a balanced diet, staying active, getting enough sleep, and managing stress.";
        } else if (lowerCaseMessage.includes('digestion')) {
          response = "Healthy digestion depends on a fiber-rich diet, adequate hydration, regular physical activity, and mindful eating habits.";
        } else if (lowerCaseMessage.includes('bone health')) {
          response = "To support bone health, consume adequate calcium and vitamin D, engage in weight-bearing exercises, and avoid smoking and excessive alcohol.";
        }else if (lowerCaseMessage.includes('paracetamol')) {
          response = "Paracetamol is commonly used to relieve pain and reduce fever. Always follow the recommended dosage and avoid exceeding the maximum daily limit to prevent liver damage.";
        } else if (lowerCaseMessage.includes('ibuprofen')) {
          response = "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce pain, fever, and inflammation. Take it with food to minimize stomach upset and consult a doctor if you have ulcers or kidney issues.";
        } else if (lowerCaseMessage.includes('xanax') || lowerCaseMessage.includes('alprazolam')) {
          response = "Xanax (alprazolam) is prescribed for anxiety and panic disorders. It should be used only under medical supervision due to risk of dependence and side effects. Avoid alcohol while taking this medication.";
        } else if (lowerCaseMessage.includes('antibiotics')) {
          response = "Antibiotics fight bacterial infections. It's important to complete the full prescribed course even if you feel better. Do not use antibiotics for viral infections like colds or flu.";
        } else if (lowerCaseMessage.includes('antihistamines')) {
          response = "Antihistamines help relieve allergy symptoms such as sneezing, itching, and runny nose. Some may cause drowsiness; check labels and avoid driving if affected.";
        } else if (lowerCaseMessage.includes('insulin')) {
          response = "Insulin is essential for managing blood sugar in people with diabetes. Proper storage and dosage are crucial. Always follow your healthcare provider's instructions.";
        } else if (lowerCaseMessage.includes('vaccines')) {
          response = "Vaccines protect against infectious diseases by boosting immunity. Stay up to date with recommended vaccines and consult your healthcare provider about any concerns.";
        } else if (lowerCaseMessage.includes('antidepressants')) {
          response = "Antidepressants help manage depression and anxiety disorders. They may take several weeks to work and should only be stopped under medical guidance.";
        } else if (lowerCaseMessage.includes('painkillers')) {
          response = "Painkillers relieve various types of pain. Use as directed and be cautious of potential side effects or interactions with other medications.";
        } else if (lowerCaseMessage.includes('steroids')) {
          response = "Steroids reduce inflammation and suppress immune responses. They must be used as prescribed due to potential side effects, including weight gain and mood changes.";
        }                
        else {
          response = "Thank you for your medical question. As a medical assistant, I can provide general information, but please consult with a healthcare professional for personalized medical advice. Is there anything specific about this topic you'd like to know?";
        }
      } else {
        // Respond that this is outside medical domain
        response = "I'm your AushadX medical assistant and I'm designed to help with health and medical questions only. Could you please ask me something related to healthcare, medicine, or wellness? I'd be happy to assist with those topics.";
      }
      
      return response;
    } catch (error) {
      console.error('Error processing message:', error);
      return "I'm experiencing technical difficulties. Please try again later.";
    } finally {
      setIsLoading(false);
    }
  };

  // Handle sending a message
  const handleSendMessage = async () => {
    if (inputText.trim() === '' || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      text: inputText.trim(),
      sender: 'user',
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText('');

    // Show typing indicator
    setIsLoading(true);

    // Get response from Gemini
    const aiResponse = await sendMessageToGemini(userMessage.text);

    // Add AI response to chat
    const botMessage = {
      id: (Date.now() + 1).toString(),
      text: aiResponse,
      sender: 'bot',
    };

    setMessages((prev) => [...prev, botMessage]);
    
    // Read out the AI response
    speakMessage(aiResponse);
  };

  // Text-to-speech function
  const speakMessage = (message) => {
    setIsSpeaking(true);
    Speech.speak(message, {
      language: language === 'hi' ? 'hi-IN' : 'en-US',
      pitch: 1.0,
      rate: 0.9,
      onDone: () => setIsSpeaking(false),
      onError: () => setIsSpeaking(false),
    });
  };

  // Stop speech if it's currently speaking
  const stopSpeaking = () => {
    if (isSpeaking) {
      Speech.stop();
      setIsSpeaking(false);
    }
  };

  // Handle user's voice input
  const handleVoiceInput = async (recordingUri) => {
    // Simulate speech-to-text processing
    // In a real implementation, you would call a speech-to-text service
    setIsLoading(true);
    
    await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate processing time
    
    // Generate a random medical query for demo purposes
    const sampleQueries = [
      "What are the common symptoms of a cold?",
      "How can I lower my blood pressure naturally?",
      "What should I do for a headache?",
      "How much exercise is recommended weekly?",
      "What are the symptoms of diabetes?",
      "How can I improve my sleep hygiene?",
      "What are the signs of dehydration?",
      "How much water should I drink a day?",
      "What are healthy foods for weight loss?",
      "How can I boost my immune system?",
      "How do I know if I have the flu or a cold?",
      "What are the symptoms of COVID-19?",
      "How can I relieve stress quickly?",
      "What is a normal blood pressure range?",
      "How can I manage anxiety without medication?",
      "What are some good sources of protein?",
      "What are the early signs of cancer?",
      "How often should I get a physical exam?",
      "Is intermittent fasting healthy?",
      "How can I improve my digestion naturally?",
      "What are some natural remedies for indigestion?",
      "How can I stop smoking?",
      "How many hours of sleep do adults need?",
      "Is coffee good or bad for health?",
      "What causes fatigue and low energy?",
      "How can I strengthen my heart?",
      "What are the benefits of walking daily?",
      "How do I reduce cholesterol levels naturally?",
      "What vitamins are essential for daily health?",
      "Are multivitamins necessary?",
      "How can I protect my skin from the sun?",
      "What causes acne and how can I treat it?",
      "What are the signs of vitamin D deficiency?",
      "How do I deal with seasonal allergies?",
      "What foods are good for brain health?",
      "How often should I get my eyes checked?",
      "What are healthy ways to gain weight?",
      "What are symptoms of iron deficiency?",
      "How can I prevent back pain?",
      "What is mindfulness and how can it help?",
      "Can stress cause physical illness?",
      "What are good exercises for beginners?",
      "How do I improve posture?",
      "What is the best time to work out?",
      "Are probiotics good for you?",
      "What are the dangers of processed foods?",
      "How can I make my diet more balanced?",
      "How do I treat a sore throat naturally?",
      "What are some natural ways to lower blood sugar?",
      "Is a vegetarian diet healthy?",
      "How can I increase my metabolism?",
      "What are the symptoms of a thyroid problem?",
      "How can I naturally boost my energy levels?",
      "What is the best way to lose belly fat?",
      "How do I maintain a healthy weight?",
      "What are the side effects of too much sugar?",
      "How can I reduce inflammation in the body?",
      "What are the benefits of yoga?",
      "What are the symptoms of depression?",
      "How can I support my mental health daily?",
      "What causes bloating and how can I reduce it?",
      "What are the signs of food allergies?",
      "How can I build muscle naturally?",
      "How often should I floss and brush my teeth?",
      "What are the benefits of drinking lemon water?",
      "What are the signs of high cholesterol?",
      "How can I detox my body naturally?",
      "What is the importance of fiber in the diet?",
      "What causes high blood pressure?",
      "Can lack of sleep cause weight gain?",
      "What are the benefits of drinking green tea?",
      "How can I prevent heart disease?",
      "What are common causes of joint pain?",
      "What are good habits for healthy skin?",
      "What are the effects of alcohol on the body?",
      "How do I know if I have a sleep disorder?",
      "What are the best ways to handle insomnia?",
      "What are the symptoms of a panic attack?",
      "How do I improve my lung health?",
      "What are common causes of constipation?",
      "What are the best foods for liver health?",
      "How can I stay healthy during flu season?",
      "What are the benefits of strength training?",
      "What is the role of antioxidants in health?",
      "What causes migraines and how can I treat them?",
      "How can I reduce sugar intake effectively?",
      "What are good sources of omega-3 fatty acids?",
      "How do I build a consistent workout routine?",
      "What are signs of burnout?",
      "How can I protect my mental health at work?",
      "What is a healthy resting heart rate?",
      "What are the health risks of sitting too long?",
      "What is the best way to manage chronic pain?",
      "How can I track my daily calorie intake?",
      "What are signs of calcium deficiency?",
      "How can I stay motivated to eat healthy?",
      "What should I eat before and after a workout?",
      "What are healthy snack options?",
      "How do I prevent motion sickness?",
      "What is the importance of hydration?",
      "What are common signs of aging?",
      "How can I reduce eye strain from screens?",
      "What foods help improve vision?",
      "How often should I check my cholesterol?",
      "What are the symptoms of asthma?",
      "How do I manage arthritis pain naturally?",
      "What are the best practices for dental hygiene?",
      "What are the risks of being overweight?",
      "How can I quit drinking alcohol?",
      "What are the benefits of deep breathing exercises?",
      "What causes dizziness and lightheadedness?",
      "How can I improve circulation?",
      "What are signs of poor gut health?",
      "How do I get more iron in my diet?",
      "What causes muscle cramps?",
      "What are good sleep positions for back pain?",
      "What is the link between diet and mental health?",
      "How do I boost serotonin levels naturally?",
      "What are some ways to increase daily activity?",
      "What are symptoms of sleep apnea?",
      "How can I naturally regulate hormones?",
      "What foods are high in magnesium?",
      "How can I reduce the risk of stroke?",
      "What are the signs of liver problems?",
      "What are benefits of regular stretching?",
      "What are early signs of Alzheimer’s disease?",
      "What is the impact of stress on digestion?",
      "How can I manage cravings for unhealthy food?",
      "How do I stay healthy while traveling?",
      "What are safe ways to detox after medication?",
      "What is the importance of breakfast?",
      "How can I support bone health?",
      "What causes frequent headaches?",
      "What are signs of a strong immune system?",
      "How can I reduce risk of urinary infections?",
      "What are the benefits of drinking herbal teas?",
      "What causes bad breath and how can I fix it?",
      "How do I strengthen my core muscles?",
      "What are the health benefits of turmeric?",
      "What is the ideal body fat percentage?",
      "What is the difference between good and bad fats?",
      "How can I build a healthy relationship with food?",
      "What are signs of overtraining?",
      "What are good exercises for heart health?",
      "How can I improve focus and concentration?",
      "What are the benefits of regular health checkups?",
      "What are symptoms of anemia?",
      "What is the best way to treat a fever at home?",
      "What are common causes of nausea?",
      "How do I safely fast for health reasons?",
      "What are the benefits of cold showers?",
      "How can I build mental resilience?",
      "What are signs of a healthy metabolism?",
      "How can I recover from burnout?",
      "What are natural ways to treat constipation?",
      "What is the best way to avoid jet lag?",
      "How can I support a healthy thyroid?",
      "What causes night sweats?",
      "What are signs of a hormonal imbalance?",
      "How can I prevent ear infections?",
      "What are the symptoms of a sinus infection?",
      "How do I manage seasonal depression?",
      "What foods can boost mood?",
      "What are the benefits of swimming?",
      "How can I make healthy food choices at restaurants?",
      "What causes water retention?",
      "How do I improve hand-eye coordination?",
      "What is the role of zinc in the body?",
      "What is a healthy BMI range?",
      "How do I support kidney health?",
      "What are good habits for healthy aging?",
      "How do I create a balanced meal plan?",
      "What are signs of a strong cardiovascular system?",
      "How can I stay healthy in extreme weather?",
      "What are the risks of low-carb diets?",
      "How can I strengthen my immune system after illness?",
      "What are the best ways to handle a cough?",
      "What causes brain fog?",
      "What are the symptoms of gluten intolerance?",
      "How can I stay active if I sit at a desk all day?",
      "What is the impact of blue light on sleep?",
      "How do I protect my joints during exercise?",
      "What are good bedtime habits for better sleep?",
      "What are symptoms of a vitamin B12 deficiency?",
      "How can I improve my mental clarity?",
      "What is intuitive eating?",
      "What are the benefits of regular meditation?",
      "What foods help reduce anxiety?",
      "How can I build a morning wellness routine?",
      "What is the importance of regular bowel movements?",
      "How do I recover from food poisoning?",
      "What are symptoms of lactose intolerance?",
      "How do I reduce screen time effectively?",
      "What are signs of adrenal fatigue?",
      "How can I develop better eating habits?",
      "What are the best herbs for digestion?",
      "How do I stay motivated to exercise regularly?",
      "What is the connection between hydration and skin health?",
      "What causes dark circles under the eyes?",
      "How can I support my body during menopause?",
      "What are the benefits of a plant-based diet?"
    ];
    
    const randomQuery = sampleQueries[Math.floor(Math.random() * sampleQueries.length)];
    
    // Add the transcribed text as user message
    const userMessage = {
      id: Date.now().toString(),
      text: randomQuery,
      sender: 'user',
    };
    
    setMessages((prev) => [...prev, userMessage]);
    
    // Now send the transcribed text to get AI response
    const aiResponse = await sendMessageToGemini(randomQuery);
    
    // Add AI response to chat
    const botMessage = {
      id: (Date.now() + 1).toString(),
      text: aiResponse,
      sender: 'bot',
    };
    
    setMessages((prev) => [...prev, botMessage]);
    setIsLoading(false);
    
    // Read out the AI response
    speakMessage(aiResponse);
  };

  // Start Recording
  const startRecording = async () => {
    try {
      if (isSpeaking) {
        stopSpeaking();
      }
      
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Permission to access microphone is required!');
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
      Alert.alert('Recording Error', 'Failed to start voice recording');
    }
  };

  // Stop Recording and Process Audio
  const stopRecording = async () => {
    if (!recording) return;

    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);

      // Process the recording
      handleVoiceInput(uri);
    } catch (error) {
      console.error('Error stopping recording:', error);
      Alert.alert(
        'Processing Error', 
        'There was an error processing your voice input. Please try again.'
      );
      setRecording(null);
    }
  };

  // Clear chat history
  const clearChat = () => {
    // Stop any ongoing speech
    if (isSpeaking) {
      stopSpeaking();
    }
    
    Alert.alert(
      'Clear Chat',
      'Are you sure you want to clear the chat history?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Clear',
          onPress: () => {
            setMessages([
              {
                id: '1',
                text: "Hello! I'm your AushadX medical assistant. How can I help you with your medical questions today?",
                sender: 'bot',
              },
            ]);
          },
          style: 'destructive',
        },
      ]
    );
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
          <TouchableOpacity onPress={clearChat} style={styles.clearButton}>
            <Ionicons name="trash-outline" size={22} color="white" />
          </TouchableOpacity>
        </View>

        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <TouchableOpacity 
              activeOpacity={item.sender === 'bot' ? 0.7 : 1}
              onPress={() => item.sender === 'bot' && speakMessage(item.text)}
            >
              <View style={[styles.messageBubble, item.sender === 'user' ? styles.userMessage : styles.botMessage]}>
                <Text style={styles.messageText}>{item.text}</Text>
                {item.sender === 'bot' && (
                  <TouchableOpacity 
                    style={styles.speakButton}
                    onPress={() => speakMessage(item.text)}
                  >
                    <Ionicons name="volume-medium-outline" size={16} color="rgba(255,255,255,0.7)" />
                  </TouchableOpacity>
                )}
              </View>
            </TouchableOpacity>
          )}
          contentContainerStyle={styles.messageList}
          showsVerticalScrollIndicator={false}
        />

        {isLoading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="small" color="#4466EE" />
            <Text style={styles.loadingText}>AushadX is thinking...</Text>
          </View>
        )}

        {isSpeaking && (
          <TouchableOpacity style={styles.stopSpeakingButton} onPress={stopSpeaking}>
            <Ionicons name="stop-circle" size={24} color="white" />
            <Text style={styles.stopSpeakingText}></Text>
          </TouchableOpacity>
        )}

        <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.inputContainer}>
          <View style={styles.inputRow}>
            <TouchableOpacity
              onPress={recording ? stopRecording : startRecording}
              style={[styles.voiceButton, recording && styles.recordingButton]}
              disabled={isLoading}
            >
              <Ionicons name={recording ? 'stop-circle' : 'mic'} size={24} color="white" />
            </TouchableOpacity>
            <TextInput
              style={styles.input}
              value={inputText}
              onChangeText={setInputText}
              placeholder={t('Type your medical question...')}
              placeholderTextColor="#9e9e9e"
              returnKeyType="send"
              onSubmitEditing={handleSendMessage}
              editable={!isLoading}
            />
            <TouchableOpacity
              style={[styles.sendButton, (!inputText.trim() || isLoading) && styles.disabledButton]}
              onPress={handleSendMessage}
              disabled={!inputText.trim() || isLoading}
              activeOpacity={0.7}
            >
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
  messageList: { paddingHorizontal: 16, paddingBottom: 20 },
  messageBubble: { maxWidth: '80%', padding: 12, borderRadius: 16, marginBottom: 10 },
  userMessage: { alignSelf: 'flex-end', backgroundColor: '#4466EE' },
  botMessage: { alignSelf: 'flex-start', backgroundColor: '#2C2C2E' },
  messageText: { color: 'white', fontSize: 16 },
  inputContainer: { width: '100%', marginTop: 'auto', padding: 10 },
  inputRow: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#2C2C2E', borderRadius: 30, padding: 10 },
  input: { flex: 1, color: 'white', fontSize: 16, marginLeft: 10 },
  sendButton: { padding: 8 },
  disabledButton: { opacity: 0.5 },
  voiceButton: { padding: 8, backgroundColor: '#4466EE', borderRadius: 20, marginRight: 8 },
  recordingButton: { backgroundColor: '#FF4444' },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 8,
    backgroundColor: 'rgba(0,0,0,0.3)',
    borderRadius: 20,
    alignSelf: 'center',
    marginBottom: 10,
  },
  loadingText: {
    color: 'white',
    marginLeft: 8,
    fontSize: 14,
  },
  speakButton: {
    width: 40,               // Add fixed width
    height: 40,              // Add fixed height to make it a perfect circle
    borderRadius: 20,        // Half of width/height
    backgroundColor: 'rgba(68,102,238,0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 0,           // Align to the very left
    marginTop: 10,
    alignSelf: 'flex-start', // Pushes the button to the left inside its parent
  },
  stopSpeakingButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    // backgroundColor: 'rgba(255, 0, 0, 0.15)', // Light red background
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 10,
    marginTop: 10,
    marginBottom: 0,
    alignSelf: 'flex-start',
  },
  stopSpeakingText: {
    color: 'red',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 2,
    marginLeft: 4,
    alignSelf: 'flex-start',
  },
  
});

export default AIChat;