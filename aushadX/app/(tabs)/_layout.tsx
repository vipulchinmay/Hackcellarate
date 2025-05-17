import { Tabs } from "expo-router";
import React, { useMemo, useState } from "react";
import { Ionicons } from "@expo/vector-icons";
import { StyleSheet, TouchableOpacity, Text, Linking, View, Alert, Modal, FlatList } from "react-native";
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useLanguage } from "@/components/LanguageContext";
import translations from "@/components/translation";
import { title } from "process";

export default function TabLayout() {
  const [darkMode, setDarkMode] = useState(true);
  const [showLanguageModal, setShowLanguageModal] = useState(false);
  const { language, changeLanguage } = useLanguage();

  // Load theme preference when component mounts
  React.useEffect(() => {
    const loadThemePreference = async () => {
      try {
        const savedTheme = await AsyncStorage.getItem('darkMode');
        if (savedTheme !== null) {
          setDarkMode(savedTheme === 'true');
        }
      } catch (error) {
        console.error('Error loading theme:', error);
      }
    };
    loadThemePreference();
  }, []);

  const toggleTheme = async () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    try {
      await AsyncStorage.setItem('darkMode', newMode.toString());
    } catch (error) {
      console.error('Error saving theme:', error);
    }
  };

  const selectLanguage = async (lang: string) => {
    await changeLanguage(lang);
    setShowLanguageModal(false);
  };

  const t = (key: string) => translations[language]?.[key] || key;

  const screenOptions = useMemo(
    () => ({
      tabBarActiveTintColor: "#f84040",
      tabBarInactiveTintColor: "#bbb",
      tabBarStyle: darkMode ? styles.darkTabBar : styles.lightTabBar,
      tabBarLabelStyle: styles.tabBarLabel,
      headerStyle: darkMode ? styles.darkHeader : styles.lightHeader,
      headerShadowVisible: false,
      headerTintColor: darkMode ? "#fff" : "#f56e7d",
      headerTitle: t("AushadX"),
      headerLeft: () => <SOSButton t={t} />,
      headerRight: () => (
        <View style={{ flexDirection: "row", alignItems: "center" }}>
          <LanguageButton 
            openModal={() => setShowLanguageModal(true)} 
            currentLanguage={language}
          />
          <ThemeToggleButton darkMode={darkMode} toggleTheme={toggleTheme} />
        </View>
      ),
    }),
    [darkMode, language, t]
  );

  const tabScreens = [
    {
      name: "index",
      title: "Home",
      icon: { focused: "home-sharp", outline: "home-outline" }
    },
    {
      name: "profile",
      title: "Profile",
      icon: { focused: "person-circle", outline: "person-circle-outline" }
    },
    {
      name: "DoctorConsult",
      title: "Doctor Consult",
      icon: { focused: "medkit", outline: "medkit-outline" }
    },
    {
      name: "Schedule",
      title: "Schedule",
      icon: { focused: "calendar", outline: "calendar-outline" }
    }
    // {
    //   name:"AIChat",
    //   title: "AIChat",
    //   icon:{ focused:"chatbubble", outline: "chatbubble-outline"}
    // }
  ];

  return (
    <>
      <Tabs screenOptions={screenOptions}>
        {tabScreens.map((screen) => (
          <Tabs.Screen
            key={screen.name}
            name={screen.name}
            options={{
              headerTitle: t(screen.title),
              tabBarLabel: t(screen.title),
              tabBarIcon: ({ focused, color }) => (
                <Ionicons 
                  name={focused ? screen.icon.focused : screen.icon.outline} 
                  color={color} 
                  size={26} 
                />
              ),
            }}
          />
        ))}
      </Tabs>

      <Modal 
        visible={showLanguageModal} 
        transparent 
        animationType="fade"
        onRequestClose={() => setShowLanguageModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={[
            styles.modalContent,
            darkMode && styles.darkModalContent
          ]}>
            <Text style={[
              styles.modalTitle,
              darkMode && styles.darkText
            ]}>
              {t("Select Your Language")}
            </Text>
            <FlatList
              data={Object.keys(translations)}
              keyExtractor={(item) => item}
              renderItem={({ item }) => (
                <TouchableOpacity 
                  onPress={() => selectLanguage(item)} 
                  style={[
                    styles.languageOption,
                    language === item && styles.selectedLanguage,
                    darkMode && styles.darkLanguageOption
                  ]}
                >
                  <Text style={[
                    styles.languageText,
                    language === item && styles.selectedLanguageText,
                    darkMode && styles.darkText
                  ]}>
                    {translations[item]?.languageName || item}
                  </Text>
                </TouchableOpacity>
              )}
            />
            <TouchableOpacity 
              onPress={() => setShowLanguageModal(false)} 
              style={styles.closeButton}
            >
              <Text style={styles.closeButtonText}>{t("Close")}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </>
  );
}

const LanguageButton = ({ openModal, currentLanguage }) => (
  <TouchableOpacity onPress={openModal} style={styles.languageButton}>
    <Text style={styles.languageButtonText}>
      {'🌍'}
    </Text>
  </TouchableOpacity>
);

const SOSButton = ({ t }) => {
  const handleSOSPress = () => {
    Alert.alert(
      t("Emergency Call"),
      t("Do you want to call 108 for emergency services?"),
      [
        { text: t("Cancel"), style: "cancel" },
        { text: t("Call"), onPress: () => Linking.openURL("tel:108") },
      ]
    );
  };

  return (
    <TouchableOpacity onPress={handleSOSPress} style={styles.sosButton}>
      <Text style={styles.sosText}>SOS</Text>
    </TouchableOpacity>
  );
};

const ThemeToggleButton = ({ darkMode, toggleTheme }) => (
  <TouchableOpacity 
    onPress={toggleTheme} 
    style={[styles.toggleButton, darkMode ? styles.darkToggle : styles.lightToggle]}
  >
    <Text style={[styles.toggleText, darkMode ? { color: "#fff" } : { color: "#25292e" }]}>
      {darkMode ? "☀️" : "🌙"}
    </Text>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  // ... your existing styles ...
  darkModalContent: {
    backgroundColor: '#25292e',
  },
  darkText: {
    color: '#fff',
  },
  darkLanguageOption: {
    borderBottomColor: '#444',
  },
  selectedLanguage: {
    backgroundColor: '',
  },
  selectedLanguageText: {
    color: '',
    fontWeight: 'bold',
  },
  lightHeader: { backgroundColor: "#f5f5f5", shadowColor: "#000", shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 4 },
  darkHeader: { backgroundColor: "#25292e", shadowColor: "#000", shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 4 },
  lightTabBar: { backgroundColor: "#fff", borderTopWidth: 0, shadowColor: "#000", shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 4 },
  darkTabBar: { backgroundColor: "#25292e", borderTopWidth: 0, shadowColor: "#000", shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 4 },
  tabBarLabel: { fontSize: 12, fontWeight: "bold" },
  sosButton: { backgroundColor: "#ff3b30", paddingVertical: 6, paddingHorizontal: 14, borderRadius: 20, marginLeft: 16 },
  sosText: { color: "#fff", fontWeight: "bold", fontSize: 14 },
  toggleButton: { paddingVertical: 6, paddingHorizontal: 16, borderRadius: 20, marginRight: 16, alignItems: "center" },
  darkToggle: { backgroundColor: "#444" },
  lightToggle: { backgroundColor: "#ddd" },
  toggleText: { fontSize: 16, fontWeight: "bold" },
  languageButton: { padding: 8, marginRight: 12 },
  languageButtonText: { fontSize: 16 },
  modalContainer: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "rgba(0,0,0,0.5)" },
  modalContent: { backgroundColor: "#fff", padding: 20, borderRadius: 10, width: 250 },
  modalTitle: { fontSize: 18, fontWeight: "bold", marginBottom: 10, textAlign: "center" },
  languageOption: { padding: 10, borderBottomWidth: 1, borderBottomColor: "#ddd" },
  languageText: { fontSize: 16, textAlign: "center" },
  closeButton: { marginTop: 10, alignSelf: "center" },
  closeButtonText: { fontSize: 16, color: "red" },
});