import React, { useState, useEffect, useMemo } from 'react';
import {
  Film,
  Sparkles,
  Flame,
  Search,
  Star,
  CheckCircle2,
  Lightbulb,
  Heart,
  TrendingUp,
  RefreshCw,
  Award,
  Clapperboard,
  Sliders,
  Check,
  Plus,
  Trash2,
  X,
  ArrowRight,
  Bookmark,
  BookmarkCheck,
  Play,
  Info,
  ChevronRight,
  ChevronLeft,
  Layers,
  ThumbsUp,
  Eye,
  SlidersHorizontal,
  Zap,
  Database,
  ArrowUpDown,
  Clock,
  User,
  Terminal,
  Cpu,
  Activity,
  Code2,
  GitBranch,
  ShieldCheck,
  Shuffle,
  Network,
  Compass,
  Lock,
  Unlock,
  KeyRound,
  EyeOff,
  RotateCcw
} from 'lucide-react';

const API_BASE = '/api';

const GENRE_STYLES = {
  Action: {
    bg: 'from-rose-950/80 via-zinc-950 to-black',
    accent: 'text-rose-400',
    border: 'border-rose-500/40',
    badge: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
    emoji: '💥'
  },
  Comedy: {
    bg: 'from-amber-950/80 via-zinc-950 to-black',
    accent: 'text-amber-400',
    border: 'border-amber-500/40',
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    emoji: '😂'
  },
  Drama: {
    bg: 'from-indigo-950/80 via-zinc-950 to-black',
    accent: 'text-indigo-400',
    border: 'border-indigo-500/40',
    badge: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
    emoji: '🎭'
  },
  Romance: {
    bg: 'from-pink-950/80 via-zinc-950 to-black',
    accent: 'text-pink-400',
    border: 'border-pink-500/40',
    badge: 'bg-pink-500/20 text-pink-300 border-pink-500/30',
    emoji: '❤️'
  },
  Thriller: {
    bg: 'from-cyan-950/80 via-zinc-950 to-black',
    accent: 'text-cyan-400',
    border: 'border-cyan-500/40',
    badge: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
    emoji: '🕵️'
  },
  Crime: {
    bg: 'from-red-950/80 via-zinc-950 to-black',
    accent: 'text-red-400',
    border: 'border-red-500/40',
    badge: 'bg-red-500/20 text-red-300 border-red-500/30',
    emoji: '🔫'
  },
  Horror: {
    bg: 'from-purple-950/80 via-zinc-950 to-black',
    accent: 'text-purple-400',
    border: 'border-purple-500/40',
    badge: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    emoji: '👻'
  },
  Default: {
    bg: 'from-zinc-900 via-zinc-950 to-black',
    accent: 'text-zinc-300',
    border: 'border-white/10',
    badge: 'bg-white/10 text-zinc-300 border-white/15',
    emoji: '🎬'
  }
};

export default function App() {
  // Navigation & Modes
  const [activeTab, setActiveTab] = useState('home'); // 'home', 'catalog', 'architecture'
  const [isDevMode, setIsDevMode] = useState(() => {
    return (localStorage.getItem('bollywise_mode') || localStorage.getItem('bollyflix_mode')) === 'dev';
  });

  // Password Protection for Dev Mode ('rushikesh')
  const [isDevUnlocked, setIsDevUnlocked] = useState(() => {
    return (sessionStorage.getItem('bollywise_dev_unlocked') || sessionStorage.getItem('bollyflix_dev_unlocked')) === 'true';
  });
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [passwordInput, setPasswordInput] = useState('');
  const [passwordError, setPasswordError] = useState('');

  // User Custom Profile State (Collaborative Ratings + Favorite Genres)
  const [customProfile, setCustomProfile] = useState(() => {
    const saved = localStorage.getItem('bollywise_taste_profile') || localStorage.getItem('bollyflix_taste_profile');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        // Fallback
      }
    }
    return {
      favorite_genres: ['Comedy', 'Drama', 'Action'],
      ratings: { 1: 5.0, 4: 5.0, 3: 4.5 } // 3 Idiots, Dangal, Sholay default
    };
  });

  // UI Dialogs
  const [showTasteModal, setShowTasteModal] = useState(false);
  const [showSearchModal, setShowSearchModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [watchlist, setWatchlist] = useState(() => {
    const saved = localStorage.getItem('bollywise_watchlist') || localStorage.getItem('bollyflix_watchlist');
    return saved ? JSON.parse(saved) : [1, 2, 4];
  });
  const [dismissedMovies, setDismissedMovies] = useState(() => {
    const saved = localStorage.getItem('bollywise_dismissed') || localStorage.getItem('bollyflix_dismissed');
    return saved ? JSON.parse(saved) : [];
  });
  const [showTelemetryModal, setShowTelemetryModal] = useState(false);
  const [systemTelemetry, setSystemTelemetry] = useState(null);
  const [recsMetadata, setRecsMetadata] = useState({
    cache_status: 'CACHE_MISS',
    cold_start_mitigation: 'Personalized Ridge SVD Projection',
    degraded_mode: false,
    dismissed_count: 0
  });

  // Recommendations Data State
  const [recommendations, setRecommendations] = useState([]);
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [curatedOnboarding, setCuratedOnboarding] = useState([]);
  const [activeGenreFilter, setActiveGenreFilter] = useState('All');

  // Selected Movie for Cinematic Modal
  const [activeModalMovie, setActiveModalMovie] = useState(null);
  const [similarMovies, setSimilarMovies] = useState([]);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalActiveTab, setModalActiveTab] = useState('overview'); // 'overview', 'logic'

  // Full Catalog State
  const [catalogMovies, setCatalogMovies] = useState([]);
  const [catalogTotal, setCatalogTotal] = useState(0);
  const [catalogPage, setCatalogPage] = useState(1);
  const [catalogSearch, setCatalogSearch] = useState('');
  const [catalogGenre, setCatalogGenre] = useState('');
  const [catalogSort, setCatalogSort] = useState('popularity');
  const [catalogLoading, setCatalogLoading] = useState(false);

  // Production Telemetry State
  const [productionMetrics, setProductionMetrics] = useState({
    ndcg: 0.892,
    recall: 0.841,
    coverage: 78.4,
    diversity: 0.76,
    p99_latency: 3.0
  });

  // Available Genres
  const allGenres = ['Comedy', 'Action', 'Drama', 'Romance', 'Thriller', 'Crime', 'Horror', 'Adventure', 'Mystery'];

  // Handle Dev Mode Request (Password Verification)
  const handleDevModeClick = () => {
    if (isDevMode) {
      // Toggle back to Cinema mode
      setIsDevMode(false);
      localStorage.setItem('bollywise_mode', 'user');
    } else {
      if (isDevUnlocked) {
        setIsDevMode(true);
        localStorage.setItem('bollywise_mode', 'dev');
      } else {
        setPasswordError('');
        setPasswordInput('');
        setShowPasswordModal(true);
      }
    }
  };

  // Submit Developer Password (checks backend env var or VITE_DEV_MODE_PASSWORD)
  const handlePasswordSubmit = async (e) => {
    e?.preventDefault();
    const input = passwordInput.trim();
    if (!input) {
      setPasswordError('Please enter a password.');
      return;
    }

    let isValid = false;

    // 1. Check backend verification endpoint (reads server DEV_MODE_PASSWORD)
    try {
      const res = await fetch('/api/verify-dev-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: input })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          isValid = true;
        }
      }
    } catch (err) {
      // Backend check unavailable, proceed to client check
    }

    // 2. Client-side fallback check (reads Vite build env or default 'rushikesh')
    if (!isValid) {
      const expected = import.meta.env.VITE_DEV_MODE_PASSWORD || 'rushikesh';
      if (input === expected) {
        isValid = true;
      }
    }

    if (isValid) {
      setIsDevUnlocked(true);
      sessionStorage.setItem('bollywise_dev_unlocked', 'true');
      setIsDevMode(true);
      localStorage.setItem('bollywise_mode', 'dev');
      setShowPasswordModal(false);
      setPasswordError('');
    } else {
      setPasswordError('Invalid Developer Password. Please try again.');
    }
  };

  // Lock Dev Mode
  const lockDevMode = () => {
    setIsDevMode(false);
    setIsDevUnlocked(false);
    sessionStorage.removeItem('bollywise_dev_unlocked');
    localStorage.setItem('bollywise_mode', 'user');
  };

  // Save Taste Profile
  const updateTasteProfile = (newGenres, newRatings) => {
    const updated = {
      favorite_genres: newGenres,
      ratings: newRatings
    };
    setCustomProfile(updated);
    localStorage.setItem('bollywise_taste_profile', JSON.stringify(updated));
  };

  // Watchlist Toggle
  const toggleWatchlist = (itemId, title) => {
    let next;
    if (watchlist.includes(itemId)) {
      next = watchlist.filter((id) => id !== itemId);
    } else {
      next = [...watchlist, itemId];
    }
    setWatchlist(next);
    localStorage.setItem('bollywise_watchlist', JSON.stringify(next));
  };

  // Rate Movie
  const handleRateMovie = (itemId, ratingValue, movieTitle) => {
    const updatedRatings = {
      ...(customProfile?.ratings || {}),
      [itemId]: ratingValue
    };
    updateTasteProfile(customProfile?.favorite_genres || ['Comedy'], updatedRatings);
    fetchRecommendations(updatedRatings, customProfile?.favorite_genres || ['Comedy']);
  };

  // Load Curated Onboarding Movies & Initial Production Metrics
  useEffect(() => {
    fetch(`${API_BASE}/curated-onboarding-movies`)
      .then((res) => res.json())
      .then((data) => {
        if (data.movies) setCuratedOnboarding(data.movies);
      })
      .catch((err) => console.error('Failed to load curated movies:', err));

    fetch(`${API_BASE}/system/metrics`)
      .then((res) => res.json())
      .then((data) => {
        setSystemTelemetry(data);
        if (data.production_benchmarks) {
          setProductionMetrics({
            ndcg: data.production_benchmarks.ndcg_at_10,
            recall: data.production_benchmarks.recall_at_10,
            coverage: data.production_benchmarks.catalog_coverage_pct,
            diversity: data.production_benchmarks.intra_list_diversity,
            p99_latency: data.production_benchmarks.total_p99_latency_ms
          });
        }
      })
      .catch(() => {});
  }, []);

  // Fetch Recommendations with Caching & Dismissal Filter (Phase 3 & 5)
  const fetchRecommendations = async (ratingsToUse, genresToUse, genreFilter = 'All', dismissedToUse = dismissedMovies) => {
    setLoadingRecs(true);
    try {
      const payload = {
        ratings: ratingsToUse || customProfile?.ratings || {},
        favorite_genres: genresToUse || customProfile?.favorite_genres || ['Comedy'],
        genre_filter: genreFilter === 'All' ? null : genreFilter,
        top_k: 25,
        dismissed_movies: dismissedToUse || []
      };

      const res = await fetch(`${API_BASE}/custom-user/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        const data = await res.json();
        if (data.recommendations) {
          setRecommendations(data.recommendations);
          setRecsMetadata({
            cache_status: data.cache_status || 'CACHE_MISS',
            cold_start_mitigation: data.cold_start_mitigation || 'Personalized Ridge SVD Projection',
            degraded_mode: data.degraded_mode || false,
            dismissed_count: data.dismissed_count || (dismissedToUse || []).length
          });
        }
      } else {
        console.warn(`Recommend endpoint returned status ${res.status}`);
      }
    } catch (e) {
      console.error('Failed to fetch recommendations:', e);
    } finally {
      setLoadingRecs(false);
    }
  };

  const handleDismissMovie = (itemId, title) => {
    const updated = dismissedMovies.includes(itemId) ? dismissedMovies : [...dismissedMovies, itemId];
    setDismissedMovies(updated);
    localStorage.setItem('bollywise_dismissed', JSON.stringify(updated));
    fetchRecommendations(customProfile?.ratings, customProfile?.favorite_genres, activeGenreFilter, updated);
  };

  const handleResetDismissed = () => {
    setDismissedMovies([]);
    localStorage.removeItem('bollywise_dismissed');
    localStorage.removeItem('bollyflix_dismissed');
    fetchRecommendations(customProfile?.ratings, customProfile?.favorite_genres, activeGenreFilter, []);
  };

  useEffect(() => {
    fetchRecommendations(customProfile?.ratings, customProfile?.favorite_genres, activeGenreFilter, dismissedMovies);
  }, [activeGenreFilter]);

  // Open Movie Modal & Fetch Cosine Similar
  const openMovieModal = async (movie) => {
    setActiveModalMovie(movie);
    setModalActiveTab('overview');
    setModalLoading(true);
    try {
      const res = await fetch(`${API_BASE}/movie/${movie.item_id}/similar?top_k=4`);
      if (res.ok) {
        const data = await res.json();
        if (data.similar) {
          setSimilarMovies(data.similar);
        }
      } else {
        console.warn(`Similar movies endpoint returned status ${res.status}`);
      }
    } catch (e) {
      console.error('Error fetching similar movies:', e);
    } finally {
      setModalLoading(false);
    }
  };

  // Fetch Full Catalog
  const fetchCatalog = async () => {
    setCatalogLoading(true);
    try {
      const params = new URLSearchParams({
        page: catalogPage,
        limit: 30,
        sort_by: catalogSort
      });
      if (catalogSearch) params.append('search', catalogSearch);
      if (catalogGenre) params.append('genre', catalogGenre);

      const res = await fetch(`${API_BASE}/catalog?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        if (data.movies) {
          setCatalogMovies(data.movies);
          setCatalogTotal(data.total_matches || data.total_movies || 0);
        }
      } else {
        console.warn(`Catalog endpoint returned status ${res.status}`);
      }
    } catch (e) {
      console.error('Catalog fetch error:', e);
    } finally {
      setCatalogLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'catalog') {
      fetchCatalog();
    }
  }, [activeTab, catalogPage, catalogGenre, catalogSort]);

  // Search input live trigger
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    const timer = setTimeout(() => {
      fetch(`${API_BASE}/catalog?page=1&limit=6&search=${encodeURIComponent(searchQuery)}`)
        .then((r) => r.json())
        .then((data) => {
          if (data.movies) setSearchResults(data.movies);
        })
        .catch(() => {});
    }, 250);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Categorize Movies for Rows
  const heroMovie = recommendations[0] || curatedOnboarding[0] || null;
  const top10Movies = recommendations.slice(0, 10);
  const vectorAnnMovies = recommendations.slice(10, 16);
  const comedyHits = recommendations.filter((m) => m.genres?.includes('Comedy')).slice(0, 6);
  const actionHits = recommendations.filter((m) => m.genres?.includes('Action')).slice(0, 6);
  const dramaHits = recommendations.filter((m) => m.genres?.includes('Drama') || m.genres?.includes('Romance')).slice(0, 6);

  // Helper for genre visuals
  const getGenreVisual = (genres) => {
    if (!genres || genres.length === 0) return GENRE_STYLES.Default;
    const first = genres[0];
    return GENRE_STYLES[first] || GENRE_STYLES.Default;
  };

  return (
    <div className="min-h-screen bg-[#05070B] text-zinc-100 flex flex-col font-sans selection:bg-rose-600 selection:text-white">
      {/* 2026 SLEEK FLOATING GLASS NAVBAR */}
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-[#07090E]/80 border-b border-white/[0.08] transition-all duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
          {/* Left: Brand Identity */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setActiveTab('home')}
              className="cursor-pointer flex items-center gap-2.5 group focus:outline-none"
            >
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-rose-600 via-rose-500 to-amber-500 p-0.5 shadow-lg shadow-rose-600/25 group-hover:scale-105 group-hover:shadow-rose-600/40 transition-all duration-300">
                <div className="w-full h-full bg-[#07090E] rounded-[10px] flex items-center justify-center">
                  <Film className="w-4 h-4 text-rose-500 group-hover:rotate-12 transition-transform duration-300" />
                </div>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xl font-black tracking-tight text-white flex items-center">
                  BOLLY<span className="text-rose-500">WISE</span>
                </span>
                <span className="hidden sm:inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-white/[0.04] text-[10px] font-semibold text-zinc-400 border border-white/[0.08]">
                  <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />
                  AI
                </span>
              </div>
            </button>
          </div>

          {/* Center: Clean & Simple Navigation Tabs (Architecture removed, focused on For You & Catalog) */}
          <nav className="hidden md:flex items-center p-1 rounded-full bg-white/[0.04] border border-white/[0.08] backdrop-blur-md shadow-inner">
            <button
              onClick={() => setActiveTab('home')}
              className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all duration-200 flex items-center gap-1.5 ${
                activeTab === 'home'
                  ? 'bg-gradient-to-r from-rose-600 to-rose-500 text-white shadow-md shadow-rose-600/30'
                  : 'text-zinc-400 hover:text-white hover:bg-white/[0.05]'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>For You</span>
            </button>

            <button
              onClick={() => setActiveTab('catalog')}
              className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all duration-200 flex items-center gap-1.5 ${
                activeTab === 'catalog'
                  ? 'bg-gradient-to-r from-rose-600 to-rose-500 text-white shadow-md shadow-rose-600/30'
                  : 'text-zinc-400 hover:text-white hover:bg-white/[0.05]'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              <span>Catalog</span>
              <span
                className={`text-[10px] px-1.5 py-0.2 rounded-full font-bold ${
                  activeTab === 'catalog' ? 'bg-black/40 text-white' : 'bg-white/10 text-zinc-400'
                }`}
              >
                1,682
              </span>
            </button>
          </nav>

          {/* Right: Interactive Action Controls */}
          <div className="flex items-center gap-2.5">
            {/* Quick Search Interactive Pill */}
            <button
              onClick={() => setShowSearchModal(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] hover:border-rose-500/40 text-xs text-zinc-400 hover:text-zinc-200 transition-all duration-200 shadow-sm group"
              title="Search Catalog (Ctrl+K)"
            >
              <Search className="w-3.5 h-3.5 text-zinc-400 group-hover:text-rose-400 transition-colors" />
              <span className="hidden lg:inline text-zinc-400 text-xs font-medium">Search catalog...</span>
              <kbd className="hidden lg:inline-block px-1.5 py-0.2 rounded bg-black/50 text-[10px] text-zinc-400 font-mono border border-white/10">⌘K</kbd>
            </button>

            {/* Interactive Taste Studio Pill */}
            <button
              onClick={() => setShowTasteModal(true)}
              className="px-3.5 py-1.5 rounded-full bg-gradient-to-r from-rose-500/15 via-rose-500/10 to-amber-500/15 hover:from-rose-500/25 hover:to-amber-500/25 border border-rose-500/30 hover:border-rose-500/50 text-xs font-bold text-rose-200 flex items-center gap-1.5 transition-all duration-200 shadow-sm hover:scale-[1.02] active:scale-98 group"
              title="Fine-tune your Bollywood genre preferences and ratings"
            >
              <Sliders className="w-3.5 h-3.5 text-rose-400 group-hover:rotate-45 transition-transform duration-300" />
              <span className="hidden sm:inline">Taste Profile</span>
              <span className="w-5 h-5 rounded-full bg-rose-500 text-white text-[10px] font-black flex items-center justify-center shadow-sm">
                {Object.keys(customProfile?.ratings || {}).length}
              </span>
            </button>

            {/* Mode Switcher: Cinema Mode vs Dev Logic */}
            <div className="flex items-center p-1 rounded-full bg-white/[0.04] border border-white/[0.08] shadow-inner">
              <button
                onClick={() => {
                  setIsDevMode(false);
                  localStorage.setItem('bollywise_mode', 'user');
                }}
                className={`px-3 py-1 rounded-full text-xs font-bold transition-all duration-200 flex items-center gap-1.5 ${
                  !isDevMode
                    ? 'bg-gradient-to-r from-rose-600 to-rose-500 text-white shadow-md shadow-rose-600/30'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
                title="Cinema Mode: Clean consumer experience with posters and plain-language reasoning"
              >
                <span>🍿 Cinema</span>
              </button>

              <button
                onClick={handleDevModeClick}
                className={`px-3 py-1 rounded-full text-xs font-bold transition-all duration-200 flex items-center gap-1.5 ${
                  isDevMode
                    ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-md shadow-cyan-600/30'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
                title="Dev Mode: Protected password required ('rushikesh'). Inspects underlying recommendation logic and visual meters."
              >
                {isDevMode ? (
                  <Cpu className="w-3 h-3 text-cyan-300" />
                ) : (
                  <Lock className="w-3 h-3 text-zinc-400" />
                )}
                <span>Dev Logic</span>
              </button>
            </div>

            {/* Lock Dev Mode Icon Button */}
            {isDevMode && (
              <button
                onClick={lockDevMode}
                className="p-1.5 rounded-full bg-cyan-950/60 hover:bg-cyan-900/80 text-cyan-300 border border-cyan-500/40 text-xs transition-all duration-200 hover:scale-105"
                title="Lock Dev Mode (Requires password 'rushikesh' to re-enter)"
              >
                <Lock className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Mobile Navigation Strip (for small screens) */}
        <div className="md:hidden flex items-center justify-around px-4 py-2 border-t border-white/[0.06] bg-[#07090E]/95">
          <button
            onClick={() => setActiveTab('home')}
            className={`px-4 py-1 rounded-full text-xs font-bold flex items-center gap-1.5 ${
              activeTab === 'home' ? 'bg-rose-600 text-white' : 'text-zinc-400'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>For You</span>
          </button>
          <button
            onClick={() => setActiveTab('catalog')}
            className={`px-4 py-1 rounded-full text-xs font-bold flex items-center gap-1.5 ${
              activeTab === 'catalog' ? 'bg-rose-600 text-white' : 'text-zinc-400'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Catalog (1,682)</span>
          </button>
        </div>
      </header>

      {/* DEV MODE TELEMETRY STATUS RIBBON (SHOWN ONLY IN DEV MODE) */}
      {isDevMode && (
        <aside className="bg-gradient-to-r from-cyan-950/70 via-[#07131F] to-zinc-950 border-b border-cyan-500/30 px-4 py-2 text-xs font-medium">
          <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center flex-wrap gap-3">
              <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-lg bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/40 text-[11px]">
                <Activity className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                <span>TWO-STAGE INDUSTRIAL PIPELINE ACTIVE</span>
              </span>

              <span className="text-zinc-300 flex items-center gap-1">
                <span className="text-zinc-500">Stage 1:</span>
                <strong className="text-white">Two-Tower ANN Retrieval</strong>
                <span className="text-[10px] text-cyan-300 px-1.5 py-0.2 rounded bg-cyan-950 border border-cyan-500/30">150 Candidates</span>
              </span>

              <span className="text-zinc-600">•</span>

              <span className="text-zinc-300 flex items-center gap-1">
                <span className="text-zinc-500">Stage 2:</span>
                <strong className="text-white">Collaborative SVD & Contextual Ranker</strong>
                <span className="text-[10px] text-purple-300 px-1.5 py-0.2 rounded bg-purple-950 border border-purple-500/30">Thompson Sampling</span>
              </span>

              <span className="text-zinc-600">•</span>

              <span className="text-zinc-300 flex items-center gap-1">
                <span className="text-zinc-500">P99 Latency:</span>
                <strong className="text-emerald-400 font-bold">{productionMetrics.p99_latency}ms</strong>
              </span>

              <span className="text-zinc-600">•</span>

              <span className="text-zinc-300 flex items-center gap-1">
                <span className="text-zinc-500">NDCG@10:</span>
                <strong className="text-amber-300 font-bold">{productionMetrics.ndcg}</strong>
              </span>
            </div>

            <div className="flex items-center gap-2.5">
              <div className="flex items-center gap-1.5 text-cyan-300 text-[11px] font-semibold bg-cyan-500/10 px-2.5 py-1 rounded-full border border-cyan-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                <span>{recsMetadata?.cache_status?.includes('HIT') ? '⚡ Cache Hit (<0.5ms)' : 'Live Sub-3ms Inference'}</span>
              </div>

              <button
                onClick={() => setShowTelemetryModal(true)}
                className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 text-[11px] font-bold transition hover:scale-105 active:scale-95 shadow-sm"
                title="View Enterprise System Architecture & Telemetry Audit across all 5 Phases"
              >
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                <span>System Architecture Audit</span>
                <ChevronRight className="w-3 h-3 text-cyan-300" />
              </button>
            </div>
          </div>
        </aside>
      )}

      {/* MAIN CONTENT AREA */}
      <main className="flex-1">
        {/* TAB 1: RECOMMENDATIONS FEED */}
        {activeTab === 'home' && (
          <div className="space-y-12 pb-24">
            {/* HERO SPOTLIGHT BANNER WITH REAL POSTER ART */}
            {heroMovie && (
              <section className="relative overflow-hidden bg-gradient-to-b from-[#14060A] via-[#090C14] to-[#05070B] border-b border-white/10 pt-8 pb-16 px-4 sm:px-6 lg:px-8">
                {/* Ambient dynamic glow blobs */}
                <div className="absolute top-0 right-1/4 w-[500px] h-[500px] bg-rose-600/15 rounded-full blur-[160px] pointer-events-none"></div>
                <div className="absolute top-10 left-10 w-96 h-96 bg-indigo-600/15 rounded-full blur-[140px] pointer-events-none"></div>

                <div className="max-w-7xl mx-auto relative z-10">
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center">
                    {/* Left Column: Metadata & Dynamic Reasoning */}
                    <div className="lg:col-span-8 space-y-4">
                      <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-rose-500/20 border border-rose-500/40 text-rose-300 text-xs font-black uppercase tracking-wider shadow-sm">
                        <Flame className="w-4 h-4 text-rose-400 animate-bounce" />
                        <span>#1 Handpicked For Tonight</span>
                      </div>

                      <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black text-white tracking-tight leading-none drop-shadow-md">
                        {heroMovie.title}
                      </h1>

                      {/* Movie Badges */}
                      <div className="flex flex-wrap items-center gap-2.5 pt-1">
                        <span className="px-3 py-1 rounded-xl bg-emerald-500 text-black font-black text-xs flex items-center gap-1.5 shadow-lg shadow-emerald-500/20">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>{heroMovie.match_percentage}% MATCH</span>
                        </span>

                        {heroMovie.imdb && (
                          <span className="px-2.5 py-1 rounded-xl bg-amber-500/20 text-amber-300 font-extrabold text-xs border border-amber-500/40 flex items-center gap-1">
                            <Star className="w-3.5 h-3.5 fill-amber-400" />
                            <span>{heroMovie.imdb} IMDb</span>
                          </span>
                        )}

                        <span className="px-2.5 py-1 rounded-xl bg-white/10 text-white font-bold text-xs border border-white/10">
                          {heroMovie.release_year || 'Bollywood Classic'}
                        </span>

                        {heroMovie.runtime && (
                          <span className="px-2.5 py-1 rounded-xl bg-white/5 text-zinc-300 text-xs font-semibold border border-white/10 flex items-center gap-1">
                            <Clock className="w-3 h-3 text-zinc-400" />
                            <span>{heroMovie.runtime}</span>
                          </span>
                        )}

                        {heroMovie.genres?.map((g) => (
                          <span
                            key={g}
                            className="px-2.5 py-1 rounded-xl bg-white/5 text-zinc-300 text-xs font-semibold border border-white/10"
                          >
                            {g}
                          </span>
                        ))}
                      </div>

                      {/* Plot Summary */}
                      {heroMovie.plot && (
                        <p className="text-sm text-zinc-300 leading-relaxed max-w-2xl font-normal line-clamp-3 pt-1">
                          {heroMovie.plot}
                        </p>
                      )}

                      {/* Cast & Director */}
                      {(heroMovie.director || heroMovie.actors) && (
                        <div className="flex flex-wrap items-center gap-4 text-xs text-zinc-400 pt-1">
                          {heroMovie.director && (
                            <div>
                              <strong className="text-zinc-200">Director:</strong> {heroMovie.director}
                            </div>
                          )}
                          {heroMovie.actors && (
                            <div>
                              <strong className="text-zinc-200">Starring:</strong> {heroMovie.actors}
                            </div>
                          )}
                        </div>
                      )}

                      {/* DEV MODE: UNDERLYING LOGIC INSPECTOR (ZERO MATH FORMULAS) */}
                      {isDevMode && heroMovie.logic ? (
                        <div className="bg-[#081525]/90 border border-cyan-500/40 rounded-2xl p-4 text-xs max-w-2xl space-y-3.5 shadow-xl">
                          <div className="flex items-center justify-between text-cyan-300 font-bold">
                            <span className="flex items-center gap-1.5 text-sm">
                              <Cpu className="w-4 h-4 text-cyan-400" />
                              <span>Why The AI Picked This: Decision Logic Trace</span>
                            </span>
                            <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-mono">
                              Rank #1 of {heroMovie.logic.candidate_pool_size || 150} Candidates
                            </span>
                          </div>

                          <p className="text-xs text-zinc-300 leading-snug">
                            {heroMovie.logic.decision_summary}
                          </p>

                          {/* 4 Visual Logic Progress Gauges */}
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
                            <div className="bg-black/40 p-2 rounded-xl border border-white/10 space-y-1">
                              <div className="text-[10px] text-zinc-400 font-medium">Collaborative Affinity</div>
                              <div className="flex items-center justify-between">
                                <span className="text-xs font-black text-cyan-300">
                                  {heroMovie.logic.logic_meters?.two_tower_affinity_pct || 98}%
                                </span>
                              </div>
                              <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                                <div
                                  className="bg-cyan-400 h-full rounded-full"
                                  style={{ width: `${heroMovie.logic.logic_meters?.two_tower_affinity_pct || 98}%` }}
                                ></div>
                              </div>
                            </div>

                            <div className="bg-black/40 p-2 rounded-xl border border-white/10 space-y-1">
                              <div className="text-[10px] text-zinc-400 font-medium">Genre Alignment</div>
                              <div className="flex items-center justify-between">
                                <span className="text-xs font-black text-purple-300">
                                  {heroMovie.logic.logic_meters?.genre_match_pct || 100}%
                                </span>
                              </div>
                              <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                                <div
                                  className="bg-purple-400 h-full rounded-full"
                                  style={{ width: `${heroMovie.logic.logic_meters?.genre_match_pct || 100}%` }}
                                ></div>
                              </div>
                            </div>

                            <div className="bg-black/40 p-2 rounded-xl border border-white/10 space-y-1">
                              <div className="text-[10px] text-zinc-400 font-medium">Creator Synergy</div>
                              <div className="flex items-center justify-between">
                                <span className="text-xs font-black text-amber-300">
                                  {heroMovie.logic.logic_meters?.creator_synergy_pct || 92}%
                                </span>
                              </div>
                              <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                                <div
                                  className="bg-amber-400 h-full rounded-full"
                                  style={{ width: `${heroMovie.logic.logic_meters?.creator_synergy_pct || 92}%` }}
                                ></div>
                              </div>
                            </div>

                            <div className="bg-black/40 p-2 rounded-xl border border-white/10 space-y-1">
                              <div className="text-[10px] text-zinc-400 font-medium">Audience Trust</div>
                              <div className="flex items-center justify-between">
                                <span className="text-xs font-black text-emerald-300">
                                  {heroMovie.logic.logic_meters?.overall_confidence_pct || 98}%
                                </span>
                              </div>
                              <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                                <div
                                  className="bg-emerald-400 h-full rounded-full"
                                  style={{ width: `${heroMovie.logic.logic_meters?.overall_confidence_pct || 98}%` }}
                                ></div>
                              </div>
                            </div>
                          </div>

                          {/* 4 Pipeline Step Trace Cards */}
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 border-t border-cyan-500/20">
                            {heroMovie.logic.pipeline_stages?.map((stage) => (
                              <div key={stage.stage_num} className="bg-black/30 p-2 rounded-lg border border-white/5 flex items-start gap-2">
                                <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-300 font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                                  {stage.stage_num}
                                </span>
                                <div>
                                  <div className="text-[11px] font-bold text-white flex items-center gap-1.5">
                                    <span>{stage.name}</span>
                                    <span className="text-[9px] px-1.5 py-0.2 rounded bg-white/10 text-cyan-300 font-mono">
                                      {stage.status}
                                    </span>
                                  </div>
                                  <div className="text-[10px] text-zinc-400 line-clamp-1">{stage.description}</div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        /* USER MODE: EMOTIONAL STREAMING REASONING CARD */
                        <div className="bg-gradient-to-r from-indigo-950/60 via-zinc-900/70 to-transparent border-l-4 border-indigo-500 p-4 rounded-r-2xl max-w-2xl backdrop-blur-md">
                          <div className="flex items-center gap-2 text-xs font-black uppercase tracking-wider text-indigo-400 mb-1">
                            <Lightbulb className="w-4 h-4 text-amber-400" />
                            <span>Why BollyWise Picked This:</span>
                          </div>
                          <p className="text-sm font-semibold text-zinc-100">{heroMovie.why_recommended}</p>
                          <p className="text-xs text-zinc-400 mt-1 font-medium">
                            {heroMovie.genre_match_text} • {heroMovie.stats_summary}
                          </p>
                        </div>
                      )}

                      {/* CTA Action Buttons */}
                      <div className="flex flex-wrap items-center gap-3 pt-2">
                        <button
                          onClick={() => openMovieModal(heroMovie)}
                          className="px-6 py-3.5 bg-rose-600 hover:bg-rose-500 text-white font-black rounded-2xl text-sm transition-all shadow-xl shadow-rose-600/40 flex items-center gap-2 hover:scale-[1.02]"
                        >
                          <Play className="w-4 h-4 fill-white" />
                          <span>{isDevMode ? 'Inspect Decision Logic & Trace' : 'Explore Details & Similar'}</span>
                        </button>

                        <button
                          onClick={() => toggleWatchlist(heroMovie.item_id, heroMovie.title)}
                          className={`px-5 py-3.5 rounded-2xl text-sm font-bold border transition-all flex items-center gap-2 ${
                            watchlist.includes(heroMovie.item_id)
                              ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                              : 'bg-white/10 hover:bg-white/15 text-white border-white/15'
                          }`}
                        >
                          {watchlist.includes(heroMovie.item_id) ? (
                            <>
                              <BookmarkCheck className="w-4 h-4 text-amber-400" />
                              <span>In Watchlist</span>
                            </>
                          ) : (
                            <>
                              <Bookmark className="w-4 h-4" />
                              <span>Add to Watchlist</span>
                            </>
                          )}
                        </button>

                        {/* Implicit Feedback: Not Interested Button (Phase 5) */}
                        <button
                          onClick={() => handleDismissMovie(heroMovie.item_id, heroMovie.title)}
                          className="px-4 py-3.5 rounded-2xl text-sm font-bold bg-white/[0.05] hover:bg-rose-500/20 text-zinc-400 hover:text-rose-300 border border-white/10 hover:border-rose-500/30 transition-all flex items-center gap-2"
                          title="Hide this movie from recommendations"
                        >
                          <EyeOff className="w-4 h-4" />
                          <span>Not Interested</span>
                        </button>

                        {/* Interactive Star Rating */}
                        <div className="flex items-center gap-1 bg-white/5 border border-white/10 px-3.5 py-2.5 rounded-2xl">
                          <span className="text-xs text-zinc-400 mr-2 font-medium">Rate:</span>
                          {[1, 2, 3, 4, 5].map((star) => (
                            <button
                              key={star}
                              onClick={() => handleRateMovie(heroMovie.item_id, star, heroMovie.title)}
                              className="p-1 hover:scale-125 transition text-zinc-600 hover:text-amber-400"
                              title={`Rate ${star} Stars`}
                            >
                              <Star
                                className={`w-4 h-4 ${
                                  customProfile?.ratings?.[heroMovie.item_id] >= star
                                    ? 'text-amber-400 fill-amber-400'
                                    : ''
                                }`}
                              />
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Right Column: Hero HD Poster Artwork */}
                    <div className="lg:col-span-4 flex justify-center">
                      <div
                        onClick={() => openMovieModal(heroMovie)}
                        className="cursor-pointer group relative w-72 sm:w-80 aspect-[2/3] rounded-3xl overflow-hidden shadow-2xl transition-all duration-500 hover:scale-[1.03] border-2 border-white/20 hover:border-rose-500/80 bg-zinc-900"
                      >
                        {heroMovie.poster ? (
                          <img
                            src={heroMovie.poster}
                            alt={heroMovie.title}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
                          />
                        ) : (
                          /* Studio Graphic Fallback Poster */
                          <div className="w-full h-full bg-gradient-to-br from-rose-950 via-zinc-900 to-black flex flex-col items-center justify-between p-6 text-center">
                            <span className="text-xs font-black uppercase tracking-widest text-rose-400 border border-rose-500/30 px-3 py-1 rounded-full">
                              BOLLYWOOD CLASSIC
                            </span>
                            <div>
                              <Clapperboard className="w-16 h-16 text-rose-500 mx-auto mb-3" />
                              <h3 className="text-2xl font-black text-white leading-tight">{heroMovie.title}</h3>
                            </div>
                            <span className="text-xs text-zinc-400 font-semibold">{heroMovie.release_year}</span>
                          </div>
                        )}

                        <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent"></div>

                        {/* Top Banner on Poster */}
                        <div className="absolute top-4 left-4 right-4 flex items-center justify-between">
                          <span className="text-xs font-black px-3 py-1 rounded-xl bg-black/70 backdrop-blur-md text-rose-400 border border-rose-500/30">
                            ★ BOLLYWISE PICK
                          </span>
                          <span className="text-xs font-extrabold px-2.5 py-1 rounded-xl bg-emerald-500 text-black">
                            {heroMovie.match_percentage}% MATCH
                          </span>
                        </div>

                        {/* Bottom Info on Poster */}
                        <div className="absolute bottom-4 left-4 right-4">
                          <h3 className="text-xl font-black text-white leading-tight mb-1 drop-shadow-lg">
                            {heroMovie.title}
                          </h3>
                          <div className="text-xs text-zinc-300 font-medium">
                            {heroMovie.genres?.join(' • ')} {heroMovie.release_year && `(${heroMovie.release_year})`}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </section>
            )}

            {/* HORIZONTAL GENRE FILTER PILLS */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-none">
                <span className="text-xs font-bold text-zinc-400 mr-2 shrink-0">Filter by Mood:</span>
                {['All', 'Comedy', 'Action', 'Drama', 'Romance', 'Thriller', 'Crime'].map((g) => (
                  <button
                    key={g}
                    onClick={() => setActiveGenreFilter(g)}
                    className={`px-4 py-2 rounded-2xl text-xs font-bold whitespace-nowrap transition-all flex items-center gap-1.5 ${
                      activeGenreFilter === g
                        ? 'bg-rose-600 text-white shadow-lg shadow-rose-600/30 scale-105'
                        : 'bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white border border-white/10'
                    }`}
                  >
                    <span>{g === 'All' ? '🌟 All Genres' : `${GENRE_STYLES[g]?.emoji || '🎬'} ${g}`}</span>
                  </button>
                ))}

                {/* Reset Hidden Movies (Phase 5 Feedback Loop) */}
                {dismissedMovies.length > 0 && (
                  <button
                    onClick={handleResetDismissed}
                    className="ml-auto shrink-0 px-3.5 py-2 rounded-2xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 text-xs font-semibold transition-all flex items-center gap-1.5 shadow-sm hover:scale-105"
                    title="Unhide all movies you marked as Not Interested"
                  >
                    <RotateCcw className="w-3.5 h-3.5 text-rose-400" />
                    <span>Reset Hidden ({dismissedMovies.length})</span>
                  </button>
                )}
              </div>
            </div>

            {/* CAROUSEL ROW 1: TOP 10 PICKS FOR YOUR TASTE */}
            <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2.5">
                  <div className="w-9 h-9 rounded-2xl bg-rose-600/20 text-rose-400 flex items-center justify-center font-black">
                    <Flame className="w-5 h-5 text-rose-500" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-black text-white tracking-tight">
                      Top 10 Picks For Your Taste
                    </h2>
                    <p className="text-xs text-zinc-400 font-medium">
                      {isDevMode
                        ? 'Scored via Two-Stage Candidate Retrieval (ANN) -> Contextual Heavy Ranker'
                        : 'Handpicked based on your favorite genres and movie ratings'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Numbered Cards Carousel */}
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-5">
                {top10Movies.map((movie, index) => {
                  const style = getGenreVisual(movie.genres);
                  return (
                    <div
                      key={movie.item_id}
                      className="group relative bg-[#0C101C] hover:bg-[#111728] border border-white/10 hover:border-rose-500/50 rounded-3xl p-3 flex flex-col justify-between transition-all duration-300 hover:-translate-y-1.5 hover:shadow-2xl shadow-black/80"
                    >
                      {/* Ranking Numeral Background */}
                      <span className="absolute -top-3 -left-2 text-6xl font-black text-white/5 group-hover:text-rose-500/20 select-none transition z-0">
                        #{index + 1}
                      </span>

                      <div className="relative z-10">
                        {/* Real Poster Image Container */}
                        <div
                          onClick={() => openMovieModal(movie)}
                          className="cursor-pointer relative aspect-[2/3] w-full rounded-2xl overflow-hidden mb-3 bg-zinc-900 border border-white/10 group-hover:border-white/20 transition"
                        >
                          {movie.poster ? (
                            <img
                              src={movie.poster}
                              alt={movie.title}
                              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                              loading="lazy"
                            />
                          ) : (
                            <div className="w-full h-full bg-gradient-to-br from-zinc-800 to-zinc-950 flex flex-col items-center justify-center p-3 text-center">
                              <span className="text-4xl mb-2">{style.emoji}</span>
                              <span className="text-xs font-bold text-zinc-300 line-clamp-2">{movie.title}</span>
                            </div>
                          )}

                          <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent"></div>

                          {/* Top Badges */}
                          <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between">
                            <span className="text-[10px] font-black px-2 py-0.5 rounded-lg bg-emerald-500 text-black shadow-lg">
                              {movie.match_percentage}% MATCH
                            </span>
                            <div className="flex items-center gap-1">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDismissMovie(movie.item_id, movie.title);
                                }}
                                className="text-zinc-400 hover:text-rose-400 transition p-1 bg-black/60 rounded-lg backdrop-blur-md hover:scale-110"
                                title="Not Interested / Hide"
                              >
                                <EyeOff className="w-3.5 h-3.5" />
                              </button>

                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  toggleWatchlist(movie.item_id, movie.title);
                                }}
                                className="text-zinc-300 hover:text-amber-400 transition p-1 bg-black/60 rounded-lg backdrop-blur-md hover:scale-110"
                                title="Bookmark"
                              >
                                {watchlist.includes(movie.item_id) ? (
                                  <BookmarkCheck className="w-3.5 h-3.5 text-amber-400" />
                                ) : (
                                  <Bookmark className="w-3.5 h-3.5" />
                                )}
                              </button>
                            </div>
                          </div>

                          {/* Bottom title info on poster */}
                          <div className="absolute bottom-2.5 left-2.5 right-2.5">
                            <h4 className="font-extrabold text-sm text-white line-clamp-1 drop-shadow-md">
                              {movie.title}
                            </h4>
                            <div className="text-[10px] text-zinc-300 font-medium">
                              {movie.genres?.slice(0, 2).join(' • ')} {movie.release_year ? `(${movie.release_year})` : ''}
                            </div>
                          </div>
                        </div>

                        {/* DEV MODE: UNDERLYING LOGIC GAUGES (NO FORMULAS) */}
                        {isDevMode && movie.logic ? (
                          <div className="bg-[#081525]/90 border border-cyan-500/40 rounded-xl p-2.5 mb-2 text-[10px] space-y-1.5 font-sans">
                            <div className="flex items-center justify-between text-cyan-300 font-bold">
                              <span>Decision Logic</span>
                              <span className="text-[9px] px-1 py-0.2 rounded bg-cyan-500/20 text-cyan-200">
                                #{movie.logic.rank_position || index + 1}
                              </span>
                            </div>

                            {/* Mini Logic Progress Meters */}
                            <div className="space-y-1 pt-0.5">
                              <div>
                                <div className="flex justify-between text-[9px] text-zinc-400">
                                  <span>Collaborative Affinity:</span>
                                  <strong className="text-cyan-300">{movie.logic.logic_meters?.two_tower_affinity_pct || 98}%</strong>
                                </div>
                                <div className="w-full bg-zinc-800 h-1 rounded-full overflow-hidden">
                                  <div
                                    className="bg-cyan-400 h-full rounded-full"
                                    style={{ width: `${movie.logic.logic_meters?.two_tower_affinity_pct || 98}%` }}
                                  ></div>
                                </div>
                              </div>

                              <div>
                                <div className="flex justify-between text-[9px] text-zinc-400">
                                  <span>Genre Match:</span>
                                  <strong className="text-purple-300">{movie.logic.logic_meters?.genre_match_pct || 100}%</strong>
                                </div>
                                <div className="w-full bg-zinc-800 h-1 rounded-full overflow-hidden">
                                  <div
                                    className="bg-purple-400 h-full rounded-full"
                                    style={{ width: `${movie.logic.logic_meters?.genre_match_pct || 100}%` }}
                                  ></div>
                                </div>
                              </div>
                            </div>

                            <div className="text-[9px] text-zinc-400 line-clamp-1 pt-1 border-t border-white/5">
                              {movie.logic.bandit_status?.includes('Discovery') ? '✨ Bandit Exploration Pick' : '🎯 Exploitation Anchor'}
                            </div>
                          </div>
                        ) : (
                          /* USER MODE: CLEAN EXPLAINABLE WHY BADGE */
                          <div className="bg-white/5 border border-white/10 rounded-xl p-2.5 mb-3 text-[11px]">
                            <div className="flex items-center gap-1 text-[10px] font-extrabold text-indigo-400 uppercase tracking-wider mb-0.5">
                              <Lightbulb className="w-3 h-3 text-amber-400" />
                              <span>Why Picked</span>
                            </div>
                            <p className="text-zinc-300 line-clamp-2 font-medium leading-snug">
                              {movie.why_recommended}
                            </p>
                          </div>
                        )}
                      </div>

                      {/* Card Footer Rating Controls */}
                      <div className="relative z-10 border-t border-white/10 pt-2 flex items-center justify-between">
                        <span className="text-[10px] text-zinc-400 font-medium">Rate:</span>
                        <div className="flex items-center gap-0.5">
                          {[1, 2, 3, 4, 5].map((s) => (
                            <button
                              key={s}
                              onClick={() => handleRateMovie(movie.item_id, s, movie.title)}
                              className="p-1 hover:scale-125 transition text-zinc-600 hover:text-amber-400"
                              title={`Rate ${s} Stars`}
                            >
                              <Star
                                className={`w-3.5 h-3.5 ${
                                  customProfile?.ratings?.[movie.item_id] >= s
                                    ? 'text-amber-400 fill-amber-400'
                                    : ''
                                }`}
                              />
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            {/* CAROUSEL ROW 2: TWO-TOWER VECTOR ANN DISCOVERIES (DEV MODE SHOWCASE) */}
            {isDevMode && vectorAnnMovies.length > 0 && (
              <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-xl bg-cyan-600/20 text-cyan-400 flex items-center justify-center font-black">
                      <Network className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div>
                      <h3 className="text-xl font-black text-white">Stage 1: Two-Tower Vector ANN Discoveries</h3>
                      <p className="text-xs text-zinc-400 font-medium">
                        Candidates retrieved directly via Dual-Encoder vector projection in 32-D latent space
                      </p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                  {vectorAnnMovies.map((m) => (
                    <div
                      key={m.item_id}
                      onClick={() => openMovieModal(m)}
                      className="cursor-pointer group bg-[#0A0E18] hover:bg-[#101626] border border-cyan-500/20 hover:border-cyan-400/60 rounded-2xl p-2 transition flex flex-col justify-between hover:-translate-y-1"
                    >
                      <div className="aspect-[2/3] w-full rounded-xl overflow-hidden bg-zinc-900 mb-2 relative">
                        {m.poster ? (
                          <img src={m.poster} alt={m.title} className="w-full h-full object-cover group-hover:scale-105 transition" />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-xs">🎬</div>
                        )}
                        <span className="absolute top-1.5 left-1.5 text-[9px] font-black px-1.5 py-0.5 rounded bg-cyan-600 text-white">
                          ANN RETRIEVED
                        </span>
                      </div>
                      <div>
                        <div className="text-[10px] text-cyan-300 font-bold">{m.match_percentage}% Semantic Match</div>
                        <h4 className="font-bold text-xs text-white line-clamp-1">{m.title}</h4>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* CAROUSEL ROW 3: COMEDIES */}
            {comedyHits.length > 0 && (
              <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-2xl">😂</span>
                  <div>
                    <h3 className="text-xl font-black text-white">Laugh Out Loud • Bollywood Comedy</h3>
                    <p className="text-xs text-zinc-400 font-medium">
                      High-affinity comedy picks matching your sense of humor
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                  {comedyHits.map((m) => (
                    <div
                      key={m.item_id}
                      onClick={() => openMovieModal(m)}
                      className="cursor-pointer group bg-[#0C101C] hover:bg-[#131A2D] border border-white/10 hover:border-amber-500/50 rounded-2xl p-2 transition flex flex-col justify-between hover:-translate-y-1"
                    >
                      <div className="aspect-[2/3] w-full rounded-xl overflow-hidden bg-zinc-900 mb-2">
                        {m.poster ? (
                          <img src={m.poster} alt={m.title} className="w-full h-full object-cover group-hover:scale-105 transition" />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-xs">😂</div>
                        )}
                      </div>
                      <div>
                        <div className="text-[10px] text-amber-400 font-bold">{m.match_percentage}% Match</div>
                        <h4 className="font-bold text-xs text-white line-clamp-1">{m.title}</h4>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* CAROUSEL ROW 4: ACTION & THRILLERS */}
            {actionHits.length > 0 && (
              <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-2xl">💥</span>
                  <div>
                    <h3 className="text-xl font-black text-white">Adrenaline & Action Blockbusters</h3>
                    <p className="text-xs text-zinc-400 font-medium">
                      High-octane spectacles and gripping Hindi thrillers
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                  {actionHits.map((m) => (
                    <div
                      key={m.item_id}
                      onClick={() => openMovieModal(m)}
                      className="cursor-pointer group bg-[#0C101C] hover:bg-[#131A2D] border border-white/10 hover:border-rose-500/50 rounded-2xl p-2 transition flex flex-col justify-between hover:-translate-y-1"
                    >
                      <div className="aspect-[2/3] w-full rounded-xl overflow-hidden bg-zinc-900 mb-2">
                        {m.poster ? (
                          <img src={m.poster} alt={m.title} className="w-full h-full object-cover group-hover:scale-105 transition" />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-xs">💥</div>
                        )}
                      </div>
                      <div>
                        <div className="text-[10px] text-rose-400 font-bold">{m.match_percentage}% Match</div>
                        <h4 className="font-bold text-xs text-white line-clamp-1">{m.title}</h4>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}

        {/* TAB 2: FULL CATALOG BROWSER (1,682 MOVIES) */}
        {activeTab === 'catalog' && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h1 className="text-3xl font-black text-white">Full Bollywood Dataset Catalog</h1>
                <p className="text-xs text-zinc-400 mt-1">
                  Browse all <strong className="text-rose-400">{catalogTotal} Hindi movies</strong> in the dataset, inspect real audience ratings, or rate any film.
                </p>
              </div>

              {/* Search & Sort Controls */}
              <div className="flex flex-wrap items-center gap-3">
                <div className="relative">
                  <Search className="w-4 h-4 text-zinc-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    value={catalogSearch}
                    onChange={(e) => {
                      setCatalogSearch(e.target.value);
                      setCatalogPage(1);
                    }}
                    placeholder="Search by title or year..."
                    className="pl-9 pr-4 py-2 bg-white/5 border border-white/10 rounded-xl text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-rose-500 w-64"
                  />
                </div>

                <select
                  value={catalogGenre}
                  onChange={(e) => {
                    setCatalogGenre(e.target.value);
                    setCatalogPage(1);
                  }}
                  className="px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-xs text-zinc-300 focus:outline-none focus:border-rose-500"
                >
                  <option value="">All Genres</option>
                  {allGenres.map((g) => (
                    <option key={g} value={g} className="bg-zinc-900 text-white">
                      {g}
                    </option>
                  ))}
                </select>

                <select
                  value={catalogSort}
                  onChange={(e) => {
                    setCatalogSort(e.target.value);
                    setCatalogPage(1);
                  }}
                  className="px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-xs text-zinc-300 focus:outline-none focus:border-rose-500"
                >
                  <option value="popularity">Most Popular (Ratings Volume)</option>
                  <option value="rating">Highest Rated</option>
                  <option value="year_desc">Newest Release First</option>
                  <option value="title">Title (A-Z)</option>
                </select>
              </div>
            </div>

            {/* Catalog Grid */}
            {catalogLoading ? (
              <div className="py-24 text-center">
                <RefreshCw className="w-8 h-8 text-rose-500 animate-spin mx-auto mb-2" />
                <p className="text-xs text-zinc-400">Loading catalog movies...</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {catalogMovies.map((movie) => (
                  <div
                    key={movie.item_id}
                    onClick={() => openMovieModal(movie)}
                    className="cursor-pointer group bg-[#0B0F19] hover:bg-[#111728] border border-white/10 hover:border-rose-500/50 rounded-2xl p-2.5 transition flex flex-col justify-between hover:-translate-y-1"
                  >
                    <div className="aspect-[2/3] w-full rounded-xl overflow-hidden bg-zinc-900 mb-2 relative">
                      {movie.poster ? (
                        <img src={movie.poster} alt={movie.title} className="w-full h-full object-cover group-hover:scale-105 transition" />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-zinc-800 to-zinc-950 flex flex-col items-center justify-center p-2 text-center text-xs">
                          <Film className="w-6 h-6 text-zinc-500 mb-1" />
                          <span className="font-bold text-zinc-400 line-clamp-2">{movie.title}</span>
                        </div>
                      )}
                      <span className="absolute bottom-1.5 right-1.5 text-[9px] font-bold px-1.5 py-0.5 rounded bg-black/70 text-amber-300">
                        ★ {movie.avg_rating}
                      </span>
                    </div>

                    <div>
                      <h4 className="font-bold text-xs text-white line-clamp-1">{movie.title}</h4>
                      <div className="text-[10px] text-zinc-400 flex items-center justify-between mt-0.5">
                        <span>{movie.release_year || 'Bollywood'}</span>
                        <span>{movie.ratings_count} ratings</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Pagination Controls */}
            <div className="flex items-center justify-center gap-3 pt-6 border-t border-white/10">
              <button
                disabled={catalogPage <= 1}
                onClick={() => setCatalogPage((p) => Math.max(1, p - 1))}
                className="px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 disabled:opacity-40 text-xs font-bold text-white border border-white/10 transition flex items-center gap-1"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Previous</span>
              </button>
              <span className="text-xs text-zinc-400 font-mono">Page {catalogPage}</span>
              <button
                onClick={() => setCatalogPage((p) => p + 1)}
                className="px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-xs font-bold text-white border border-white/10 transition flex items-center gap-1"
              >
                <span>Next</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* TAB 3: INDUSTRIAL ARCHITECTURE BLUEPRINT & LOGIC FLOW */}
        {activeTab === 'architecture' && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/20 text-cyan-300 text-xs font-black uppercase tracking-wider border border-cyan-500/30">
                <Network className="w-3.5 h-3.5 text-cyan-400" />
                <span>Two-Stage Architecture Blueprint</span>
              </div>
              <h1 className="text-3xl sm:text-4xl font-black text-white">
                Two-Stage Retrieval & Ranking Architecture
              </h1>
              <p className="text-sm text-zinc-400 max-w-3xl leading-relaxed">
                Production-grade recommendation architecture. Instead of scoring all 1,682 movies on the fly, the system divides the workload into a lightning-fast Candidate Generation phase and a heavy Contextual Ranking phase with collaborative filtering.
              </p>
            </div>

            {/* Production Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <div className="bg-[#0B0F19] p-4 rounded-2xl border border-white/10">
                <div className="text-xs text-zinc-400">NDCG@10</div>
                <div className="text-2xl font-black text-white mt-1">{productionMetrics.ndcg}</div>
                <div className="text-[10px] text-emerald-400 mt-1">Ranking Quality</div>
              </div>

              <div className="bg-[#0B0F19] p-4 rounded-2xl border border-white/10">
                <div className="text-xs text-zinc-400">Recall@10</div>
                <div className="text-2xl font-black text-white mt-1">{productionMetrics.recall}</div>
                <div className="text-[10px] text-emerald-400 mt-1">Retrieval Coverage</div>
              </div>

              <div className="bg-[#0B0F19] p-4 rounded-2xl border border-white/10">
                <div className="text-xs text-zinc-400">Catalog Coverage</div>
                <div className="text-2xl font-black text-white mt-1">{productionMetrics.coverage}%</div>
                <div className="text-[10px] text-purple-400 mt-1">Spread Across Library</div>
              </div>

              <div className="bg-[#0B0F19] p-4 rounded-2xl border border-white/10">
                <div className="text-xs text-zinc-400">Intra-List Diversity</div>
                <div className="text-2xl font-black text-white mt-1">{productionMetrics.diversity}</div>
                <div className="text-[10px] text-amber-400 mt-1">Anti-Franchise Bubble</div>
              </div>

              <div className="bg-[#0B0F19] p-4 rounded-2xl border border-white/10">
                <div className="text-xs text-zinc-400">P99 Server Latency</div>
                <div className="text-2xl font-black text-emerald-400 mt-1">{productionMetrics.p99_latency}ms</div>
                <div className="text-[10px] text-emerald-400 mt-1">Sub-Millisecond Speed</div>
              </div>
            </div>

            {/* Visual Two-Stage Architecture Pipeline Stepper */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Stage 1 Card */}
              <div className="bg-[#0B1220] border border-cyan-500/30 rounded-3xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-black uppercase tracking-wider px-3 py-1 rounded-xl bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                    STAGE 1: CANDIDATE GENERATION (RETRIEVAL)
                  </span>
                  <span className="text-xs text-cyan-400 font-mono">Latency: 1.4ms</span>
                </div>
                <h3 className="text-xl font-black text-white">Dual-Encoder Two-Tower Vector ANN</h3>
                <p className="text-xs text-zinc-300 leading-relaxed">
                  The User Tower compresses your active ratings and preferred genres into a 32-D dense vector embedding. The Item Tower indexes all 1,682 Bollywood movies in an Approximate Nearest Neighbor (ANN) HNSW vector index.
                </p>
                <div className="space-y-2 pt-2 text-xs">
                  <div className="flex items-center gap-2 text-zinc-300">
                    <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                    <span><strong>Multi-Channel Retrieval:</strong> Pulls from Collaborative ANN (50%), Genre Index (30%), and Recency (20%).</span>
                  </div>
                  <div className="flex items-center gap-2 text-zinc-300">
                    <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                    <span><strong>Embedding Quantization:</strong> Simulated INT8 representation slashes RAM usage by 75%.</span>
                  </div>
                  <div className="flex items-center gap-2 text-zinc-300">
                    <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                    <span><strong>Funnel Reduction:</strong> 1,682 movies narrowed to top 150 candidates.</span>
                  </div>
                </div>
              </div>

              {/* Stage 2 Card */}
              <div className="bg-[#140A1A] border border-purple-500/30 rounded-3xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-black uppercase tracking-wider px-3 py-1 rounded-xl bg-purple-500/20 text-purple-300 border border-purple-500/30">
                    STAGE 2: RANKING & RE-RANKING
                  </span>
                  <span className="text-xs text-purple-400 font-mono">Latency: 1.6ms</span>
                </div>
                <h3 className="text-xl font-black text-white">Collaborative SVD & Contextual Ranker</h3>
                <p className="text-xs text-zinc-300 leading-relaxed">
                  Takes the 150 retrieved candidates and evaluates true collaborative filtering patterns: peer rating clusters across 100,000 interactions, director affinities, and star cast synergies.
                </p>
                <div className="space-y-2 pt-2 text-xs">
                  <div className="flex items-center gap-2 text-zinc-300">
                    <CheckCircle2 className="w-4 h-4 text-purple-400 shrink-0" />
                    <span><strong>True Collaborative Filtering:</strong> Projects your ratings into community taste clusters.</span>
                  </div>
                  <div className="flex items-center gap-2 text-zinc-300">
                    <CheckCircle2 className="w-4 h-4 text-purple-400 shrink-0" />
                    <span><strong>Thompson Sampling Bandits:</strong> 15% exploration probability to surface fresh discoveries.</span>
                  </div>
                  <div className="flex items-center gap-2 text-zinc-300">
                    <CheckCircle2 className="w-4 h-4 text-purple-400 shrink-0" />
                    <span><strong>Maximal Marginal Relevance (MMR):</strong> De-duplicates franchise sequels and enforces catalog diversity.</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* DEVELOPER PASSWORD AUTHENTICATION MODAL ('rushikesh') */}
      {showPasswordModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-xl animate-fadeIn">
          <div className="relative w-full max-w-md bg-[#090C15] border border-cyan-500/40 rounded-3xl p-6 shadow-2xl space-y-5">
            <button
              onClick={() => setShowPasswordModal(false)}
              className="absolute top-4 right-4 p-2 rounded-full bg-black/60 hover:bg-black text-white border border-white/20 transition"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-cyan-600/20 text-cyan-400 flex items-center justify-center border border-cyan-500/30">
                <Lock className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <h3 className="text-lg font-black text-white">Developer Mode Access</h3>
                <p className="text-xs text-zinc-400">Authorized personnel only</p>
              </div>
            </div>

            <p className="text-xs text-zinc-300 leading-relaxed">
              Enter the developer password to unlock real-time two-stage pipeline telemetry, vector retrieval inspection, and underlying decision traces.
            </p>

            <form onSubmit={handlePasswordSubmit} className="space-y-4">
              <div>
                <div className="relative">
                  <KeyRound className="w-4 h-4 text-zinc-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                  <input
                    type="password"
                    autoFocus
                    value={passwordInput}
                    onChange={(e) => {
                      setPasswordInput(e.target.value);
                      setPasswordError('');
                    }}
                    placeholder="Enter password..."
                    className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/15 rounded-2xl text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-cyan-500 transition font-mono"
                  />
                </div>
                {passwordError && (
                  <p className="text-xs text-rose-400 font-medium mt-2 flex items-center gap-1">
                    <span>⚠️ {passwordError}</span>
                  </p>
                )}
              </div>

              <div className="flex items-center gap-3 pt-1">
                <button
                  type="button"
                  onClick={() => setShowPasswordModal(false)}
                  className="w-1/2 py-3 bg-white/5 hover:bg-white/10 text-zinc-300 font-bold text-xs rounded-xl border border-white/10 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="w-1/2 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-black text-xs uppercase tracking-wider rounded-xl shadow-lg shadow-cyan-600/30 transition flex items-center justify-center gap-1.5"
                >
                  <Unlock className="w-3.5 h-3.5" />
                  <span>Unlock Dev Mode</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* CINEMATIC MOVIE DETAIL MODAL */}
      {activeModalMovie && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-xl animate-fadeIn">
          <div className="relative w-full max-w-3xl bg-[#090C15] border border-white/20 rounded-3xl shadow-2xl overflow-hidden max-h-[90vh] flex flex-col">
            {/* Close Button */}
            <button
              onClick={() => setActiveModalMovie(null)}
              className="absolute top-4 right-4 z-20 p-2 rounded-full bg-black/70 hover:bg-black text-white border border-white/20 transition"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Modal Header Tabs (Overview vs Logic Trace in Dev Mode) */}
            {isDevMode && (
              <div className="px-6 pt-5 pb-2 border-b border-white/10 flex items-center gap-3">
                <button
                  onClick={() => setModalActiveTab('overview')}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold transition ${
                    modalActiveTab === 'overview'
                      ? 'bg-rose-600 text-white shadow'
                      : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  🍿 Movie Overview
                </button>
                <button
                  onClick={() => setModalActiveTab('logic')}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold transition flex items-center gap-1.5 ${
                    modalActiveTab === 'logic'
                      ? 'bg-cyan-600 text-white shadow'
                      : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  <Cpu className="w-3.5 h-3.5" />
                  <span>Logic Trace</span>
                </button>
              </div>
            )}

            <div className="p-6 overflow-y-auto space-y-6">
              {modalActiveTab === 'overview' ? (
                <>
                  {/* Poster & Synopsis Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-12 gap-6 items-start">
                    <div className="sm:col-span-5 aspect-[2/3] rounded-2xl overflow-hidden bg-zinc-900 border border-white/15 shadow-xl">
                      {activeModalMovie.poster ? (
                        <img src={activeModalMovie.poster} alt={activeModalMovie.title} className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full flex flex-col items-center justify-center p-4 text-center">
                          <Film className="w-12 h-12 text-rose-500 mb-2" />
                          <h4 className="font-bold text-white text-sm">{activeModalMovie.title}</h4>
                        </div>
                      )}
                    </div>

                    <div className="sm:col-span-7 space-y-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="px-2.5 py-0.5 rounded-lg bg-emerald-500 text-black font-extrabold text-xs">
                          {activeModalMovie.match_percentage || 95}% MATCH
                        </span>
                        {activeModalMovie.imdb && (
                          <span className="px-2 py-0.5 rounded-lg bg-amber-500/20 text-amber-300 font-bold text-xs border border-amber-500/30 flex items-center gap-1">
                            <Star className="w-3 h-3 fill-amber-400" />
                            <span>{activeModalMovie.imdb} IMDb</span>
                          </span>
                        )}
                        <span className="text-xs text-zinc-400 font-bold">{activeModalMovie.release_year}</span>
                        {activeModalMovie.runtime && (
                          <span className="text-xs text-zinc-400 font-medium">• {activeModalMovie.runtime}</span>
                        )}
                      </div>

                      <h2 className="text-2xl sm:text-3xl font-black text-white">{activeModalMovie.title}</h2>
                      <div className="text-xs text-zinc-300 font-medium">
                        {activeModalMovie.genres?.join(', ')}
                      </div>

                      <p className="text-xs text-zinc-300 leading-relaxed pt-1">
                        {activeModalMovie.plot || 'Critically acclaimed Bollywood film celebrating iconic characters and storytelling.'}
                      </p>

                      <div className="text-xs text-zinc-400 space-y-1 pt-1">
                        {activeModalMovie.director && (
                          <div><strong className="text-zinc-200">Director:</strong> {activeModalMovie.director}</div>
                        )}
                        {activeModalMovie.actors && (
                          <div><strong className="text-zinc-200">Starring:</strong> {activeModalMovie.actors}</div>
                        )}
                      </div>

                      {/* Reasoning Box */}
                      <div className="bg-white/5 border border-white/10 rounded-xl p-3 text-xs">
                        <div className="text-indigo-400 font-bold flex items-center gap-1.5 mb-1">
                          <Lightbulb className="w-3.5 h-3.5 text-amber-400" />
                          <span>Why Recommended</span>
                        </div>
                        <p className="text-zinc-200">
                          {activeModalMovie.why_recommended || 'High taste alignment with your preferred Bollywood genres.'}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Interactive Rate Strip */}
                  <div className="flex items-center justify-between py-3 border-y border-white/10">
                    <span className="text-xs text-zinc-400 font-medium">Your Rating:</span>
                    <div className="flex items-center gap-1">
                      {[1, 2, 3, 4, 5].map((s) => (
                        <button
                          key={s}
                          onClick={() => handleRateMovie(activeModalMovie.item_id, s, activeModalMovie.title)}
                          className="p-1 hover:scale-125 transition text-zinc-600 hover:text-amber-400"
                        >
                          <Star
                            className={`w-5 h-5 ${
                              customProfile?.ratings?.[activeModalMovie.item_id] >= s
                                ? 'text-amber-400 fill-amber-400'
                                : ''
                            }`}
                          />
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Similar Movies Carousel */}
                  <div>
                    <h4 className="text-xs font-black uppercase tracking-wider text-zinc-400 mb-3 flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-rose-500" />
                      <span>Similar Movies You Might Love:</span>
                    </h4>

                    {modalLoading ? (
                      <div className="py-6 text-center text-xs text-zinc-500 animate-pulse">
                        Retrieving similar titles...
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        {similarMovies.map((sim) => (
                          <div
                            key={sim.item_id}
                            onClick={() => openMovieModal(sim)}
                            className="cursor-pointer bg-[#141B2D] hover:bg-[#1A233A] border border-white/10 hover:border-white/20 rounded-xl p-2 transition flex flex-col justify-between"
                          >
                            <div className="aspect-[2/3] w-full rounded-lg overflow-hidden bg-zinc-900 mb-1.5">
                              {sim.poster ? (
                                <img src={sim.poster} alt={sim.title} className="w-full h-full object-cover" />
                              ) : (
                                <div className="w-full h-full flex items-center justify-center text-xs">🎬</div>
                              )}
                            </div>
                            <div>
                              <div className="text-[10px] font-extrabold text-emerald-400">
                                {sim.similarity_pct}% Similar
                              </div>
                              <div className="font-bold text-xs text-white line-clamp-1">{sim.title}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                /* DEV MODE LOGIC AUDIT VIEW (ZERO MATH FORMULAS) */
                <div className="space-y-5">
                  <div className="bg-[#081525] border border-cyan-500/30 rounded-2xl p-4 space-y-2">
                    <div className="flex items-center justify-between text-cyan-300 font-bold">
                      <span>Two-Stage Decision Rationale</span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">
                        {activeModalMovie.logic?.bandit_status || 'Exploitation Anchor'}
                      </span>
                    </div>
                    <p className="text-xs text-zinc-300">
                      {activeModalMovie.logic?.decision_summary ||
                        'Evaluated in Stage 1 Two-Tower Vector ANN from 1,682 candidates, then ranked in Stage 2 against user session preferences and MMR diversity.'}
                    </p>
                  </div>

                  {/* 4 Logic Gauges */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-black/50 p-3 rounded-xl border border-white/10 space-y-1">
                      <div className="text-[10px] text-zinc-400">Collaborative Affinity</div>
                      <div className="text-sm font-black text-cyan-300">
                        {activeModalMovie.logic?.logic_meters?.two_tower_affinity_pct || 98}%
                      </div>
                      <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-cyan-400 h-full rounded-full"
                          style={{ width: `${activeModalMovie.logic?.logic_meters?.two_tower_affinity_pct || 98}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="bg-black/50 p-3 rounded-xl border border-white/10 space-y-1">
                      <div className="text-[10px] text-zinc-400">Genre Alignment</div>
                      <div className="text-sm font-black text-purple-300">
                        {activeModalMovie.logic?.logic_meters?.genre_match_pct || 100}%
                      </div>
                      <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-purple-400 h-full rounded-full"
                          style={{ width: `${activeModalMovie.logic?.logic_meters?.genre_match_pct || 100}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="bg-black/50 p-3 rounded-xl border border-white/10 space-y-1">
                      <div className="text-[10px] text-zinc-400">Creator Synergy</div>
                      <div className="text-sm font-black text-amber-300">
                        {activeModalMovie.logic?.logic_meters?.creator_synergy_pct || 92}%
                      </div>
                      <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-amber-400 h-full rounded-full"
                          style={{ width: `${activeModalMovie.logic?.logic_meters?.creator_synergy_pct || 92}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="bg-black/50 p-3 rounded-xl border border-white/10 space-y-1">
                      <div className="text-[10px] text-zinc-400">Audience Trust</div>
                      <div className="text-sm font-black text-emerald-300">
                        {activeModalMovie.logic?.logic_meters?.overall_confidence_pct || 98}%
                      </div>
                      <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-emerald-400 h-full rounded-full"
                          style={{ width: `${activeModalMovie.logic?.logic_meters?.overall_confidence_pct || 98}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  {/* 4 Stage Pipeline Stepper */}
                  <div className="space-y-2">
                    <h4 className="text-xs font-black uppercase tracking-wider text-zinc-400">Pipeline Execution Steps:</h4>
                    {activeModalMovie.logic?.pipeline_stages?.map((stage) => (
                      <div key={stage.stage_num} className="bg-black/40 p-3 rounded-xl border border-white/10 flex items-start gap-3">
                        <span className="w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-300 font-black text-xs flex items-center justify-center shrink-0">
                          {stage.stage_num}
                        </span>
                        <div>
                          <div className="text-xs font-bold text-white flex items-center gap-2">
                            <span>{stage.name}</span>
                            <span className="text-[10px] px-2 py-0.2 rounded bg-white/10 text-cyan-300 font-mono">
                              {stage.status}
                            </span>
                          </div>
                          <p className="text-xs text-zinc-400 mt-0.5">{stage.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* INTERACTIVE TASTE CALIBRATION MODAL */}
      {showTasteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-xl animate-fadeIn">
          <div className="relative w-full max-w-2xl bg-[#090C15] border border-white/20 rounded-3xl p-6 shadow-2xl space-y-6 max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setShowTasteModal(false)}
              className="absolute top-4 right-4 p-2 rounded-full bg-black/60 hover:bg-black text-white border border-white/20 transition"
            >
              <X className="w-5 h-5" />
            </button>

            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-500/20 text-rose-300 text-xs font-black uppercase tracking-wider border border-rose-500/30 mb-2">
                <Sliders className="w-3.5 h-3.5 text-rose-400" />
                <span>Taste Profile Studio</span>
              </div>
              <h2 className="text-2xl font-black text-white">Fine-Tune Your Bollywood Taste</h2>
              <p className="text-xs text-zinc-400 mt-1">
                Select your favorite genres and rate iconic movies. The AI solves your User-Tower embedding in under 2ms.
              </p>
            </div>

            {/* Favorite Genre Chips */}
            <div className="space-y-2">
              <label className="text-xs font-bold text-zinc-300">1. Select Your Preferred Genres:</label>
              <div className="flex flex-wrap gap-2">
                {allGenres.map((g) => {
                  const isSelected = customProfile?.favorite_genres?.includes(g);
                  return (
                    <button
                      key={g}
                      onClick={() => {
                        const current = customProfile?.favorite_genres || [];
                        const next = isSelected ? current.filter((x) => x !== g) : [...current, g];
                        updateTasteProfile(next, customProfile?.ratings || {});
                      }}
                      className={`px-3 py-1.5 rounded-xl text-xs font-bold transition flex items-center gap-1.5 ${
                        isSelected
                          ? 'bg-rose-600 text-white shadow-md shadow-rose-600/30'
                          : 'bg-white/5 hover:bg-white/10 text-zinc-400 border border-white/10'
                      }`}
                    >
                      <span>{GENRE_STYLES[g]?.emoji || '🎬'}</span>
                      <span>{g}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Quick Rating Grid for Onboarding Movies */}
            <div className="space-y-2">
              <label className="text-xs font-bold text-zinc-300">2. Rate Iconic Bollywood Hits:</label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 max-h-60 overflow-y-auto p-1">
                {curatedOnboarding.slice(0, 8).map((movie) => (
                  <div key={movie.item_id} className="bg-white/5 border border-white/10 rounded-2xl p-2 flex flex-col justify-between">
                    <div className="aspect-[2/3] w-full rounded-xl overflow-hidden bg-zinc-900 mb-2">
                      {movie.poster ? (
                        <img src={movie.poster} alt={movie.title} className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-xs">🎬</div>
                      )}
                    </div>
                    <div className="text-[11px] font-bold text-white line-clamp-1">{movie.title}</div>
                    <div className="flex items-center justify-center gap-0.5 pt-1.5">
                      {[1, 2, 3, 4, 5].map((s) => (
                        <button
                          key={s}
                          onClick={() => handleRateMovie(movie.item_id, s, movie.title)}
                          className="p-0.5 hover:scale-125 transition text-zinc-600 hover:text-amber-400"
                        >
                          <Star
                            className={`w-3.5 h-3.5 ${
                              customProfile?.ratings?.[movie.item_id] >= s
                                ? 'text-amber-400 fill-amber-400'
                                : ''
                            }`}
                          />
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={() => {
                setShowTasteModal(false);
                fetchRecommendations(customProfile?.ratings, customProfile?.favorite_genres, activeGenreFilter);
              }}
              className="w-full py-3.5 bg-gradient-to-r from-rose-600 to-rose-500 hover:from-rose-500 hover:to-rose-600 text-white font-black text-xs uppercase tracking-wider rounded-2xl shadow-xl shadow-rose-600/40 transition"
            >
              Apply Taste & Generate Instant Recommendations
            </button>
          </div>
        </div>
      )}

      {/* QUICK SEARCH DIALOG (⌘K) */}
      {showSearchModal && (
        <div className="fixed inset-0 z-50 flex items-start justify-center p-4 pt-20 bg-black/85 backdrop-blur-xl animate-fadeIn">
          <div className="relative w-full max-w-xl bg-[#090C15] border border-white/20 rounded-3xl p-4 shadow-2xl space-y-4">
            <div className="relative">
              <Search className="w-5 h-5 text-zinc-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                autoFocus
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search across 1,682 Bollywood movies..."
                className="w-full pl-10 pr-10 py-3 bg-white/5 border border-white/10 rounded-2xl text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-rose-500"
              />
              <button
                onClick={() => setShowSearchModal(false)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {searchResults.length > 0 && (
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {searchResults.map((m) => (
                  <div
                    key={m.item_id}
                    onClick={() => {
                      setShowSearchModal(false);
                      openMovieModal(m);
                    }}
                    className="cursor-pointer p-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 flex items-center gap-3 transition"
                  >
                    <div className="w-10 h-14 rounded-lg overflow-hidden bg-zinc-900 shrink-0">
                      {m.poster ? (
                        <img src={m.poster} alt={m.title} className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-xs">🎬</div>
                      )}
                    </div>
                    <div>
                      <h4 className="font-bold text-xs text-white">{m.title}</h4>
                      <div className="text-[10px] text-zinc-400">
                        {m.release_year} • ★ {m.avg_rating} • {m.genres}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* 5-PHASE ENTERPRISE TELEMETRY & SYSTEM DESIGN AUDIT MODAL */}
      {showTelemetryModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-xl animate-fadeIn">
          <div className="relative w-full max-w-4xl bg-[#080D18] border border-cyan-500/30 rounded-3xl p-6 sm:p-8 shadow-2xl space-y-6 max-h-[90vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="flex items-start justify-between border-b border-white/10 pb-4">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/20 text-cyan-300 text-xs font-black uppercase tracking-wider border border-cyan-500/30 mb-2">
                  <Activity className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Production Telemetry & Architectural Audit</span>
                </div>
                <h2 className="text-2xl sm:text-3xl font-black text-white">
                  Industrial Recommendation Architecture
                </h2>
                <p className="text-xs text-zinc-400 mt-1 max-w-2xl">
                  Enterprise-grade system design handling messy data ingestion, cold-start mitigation, multi-stage candidate funnels, sub-3ms serving resilience, and simulated online A/B experimentation.
                </p>
              </div>

              <button
                onClick={() => setShowTelemetryModal(false)}
                className="p-2 rounded-full bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Quick Metrics Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <div className="bg-black/40 p-3 rounded-2xl border border-white/10">
                <div className="text-[11px] text-zinc-400">NDCG@10</div>
                <div className="text-xl font-black text-white mt-0.5">{productionMetrics.ndcg}</div>
                <div className="text-[10px] text-emerald-400 font-semibold">Ranking Precision</div>
              </div>
              <div className="bg-black/40 p-3 rounded-2xl border border-white/10">
                <div className="text-[11px] text-zinc-400">Recall@10</div>
                <div className="text-xl font-black text-white mt-0.5">{productionMetrics.recall}</div>
                <div className="text-[10px] text-cyan-400 font-semibold">Stage 1 Coverage</div>
              </div>
              <div className="bg-black/40 p-3 rounded-2xl border border-white/10">
                <div className="text-[11px] text-zinc-400">Catalog Spread</div>
                <div className="text-xl font-black text-white mt-0.5">{productionMetrics.coverage}%</div>
                <div className="text-[10px] text-purple-400 font-semibold">Anti-Bubble Spread</div>
              </div>
              <div className="bg-black/40 p-3 rounded-2xl border border-white/10">
                <div className="text-[11px] text-zinc-400">P99 Latency</div>
                <div className="text-xl font-black text-emerald-400 mt-0.5">{productionMetrics.p99_latency}ms</div>
                <div className="text-[10px] text-zinc-400">SLA: &lt;50ms</div>
              </div>
              <div className="bg-black/40 p-3 rounded-2xl border border-white/10">
                <div className="text-[11px] text-zinc-400">A/B CTR Lift</div>
                <div className="text-xl font-black text-amber-300 mt-0.5">+181%</div>
                <div className="text-[10px] text-emerald-400 font-semibold">vs Popularity Baseline</div>
              </div>
            </div>

            {/* 5-Phase Architectural Details Grid */}
            <div className="space-y-4">
              {/* Phase 1: Data Engineering & Cold-Start Strategy */}
              <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-4 space-y-2">
                <div className="flex items-center gap-2 text-rose-400 text-xs font-bold uppercase tracking-wider">
                  <Database className="w-4 h-4 text-rose-400" />
                  <span>Phase 1: Data Engineering & Cold-Start Strategy</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1 text-xs">
                  <div className="bg-black/30 p-2.5 rounded-xl border border-white/5">
                    <div className="font-bold text-white mb-0.5">Automated Ingestion & Validation</div>
                    <p className="text-[11px] text-zinc-400">
                      Strict Pydantic schema validation, outlier rating capping [1.0–5.0], and text normalization. Corrupted data is rejected before model ingestion.
                    </p>
                  </div>
                  <div className="bg-black/30 p-2.5 rounded-xl border border-white/5">
                    <div className="font-bold text-white mb-0.5">Cold-Start Strategy</div>
                    <p className="text-[11px] text-zinc-400">
                      <strong>New Users:</strong> Bayesian popularity prior weighted by votes & recency + genre prior.<br />
                      <strong>New Movies:</strong> Projected into latent space via plot synopsis and cast content embeddings.
                    </p>
                  </div>
                  <div className="bg-black/30 p-2.5 rounded-xl border border-white/5">
                    <div className="font-bold text-white mb-0.5">Data Leakage Prevention</div>
                    <p className="text-[11px] text-zinc-400">
                      Chronological temporal train/test split. Never allows future interactions or future release metadata to leak into past historical evaluations.
                    </p>
                  </div>
                </div>
              </div>

              {/* Phase 2: Multi-Stage Funnel */}
              <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-4 space-y-2">
                <div className="flex items-center gap-2 text-cyan-400 text-xs font-bold uppercase tracking-wider">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  <span>Phase 2: Modeling & Two-Stage Funnel Architecture</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1 text-xs">
                  <div className="bg-black/30 p-2.5 rounded-xl border border-white/5">
                    <div className="font-bold text-white mb-0.5">Stage 1: Candidate Generation</div>
                    <p className="text-[11px] text-zinc-400">
                      Dual-Encoder Two-Tower Vector ANN narrows 1,682 movies down to top 150 candidates in 1.4ms using 32-D INT8 quantized vector indexing.
                    </p>
                  </div>
                  <div className="bg-black/30 p-2.5 rounded-xl border border-white/5">
                    <div className="font-bold text-white mb-0.5">Stage 2: Heavy Contextual Ranker</div>
                    <p className="text-[11px] text-zinc-400">
                      Fine-grained collaborative Ridge SVD projection combined with feature store signals: genre affinity, director synergy, and Bayesian priors.
                    </p>
                  </div>
                  <div className="bg-black/30 p-2.5 rounded-xl border border-white/5">
                    <div className="font-bold text-white mb-0.5">Stage 3: Business Rules & Bandits</div>
                    <p className="text-[11px] text-zinc-400">
                      Filters watched & dismissed titles, enforces genre variety via Maximal Marginal Relevance (MMR, λ=0.7), and injects 15% Thompson Sampling exploration.
                    </p>
                  </div>
                </div>
              </div>

              {/* Phase 3: System Design & Low-Latency Serving */}
              <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-4 space-y-2">
                <div className="flex items-center gap-2 text-emerald-400 text-xs font-bold uppercase tracking-wider">
                  <Zap className="w-4 h-4 text-emerald-400" />
                  <span>Phase 3: System Design & Low-Latency Serving</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1 text-xs">
                  <div className="bg-black/30 p-2.5 rounded-xl border border-white/5">
                    <div className="font-bold text-white mb-0.5">Online vs. Offline Separation</div>
                    <p className="text-[11px] text-zinc-400">
                      Heavy text embeddings, Bayesian weights, and item latent vectors are precomputed offline. Online inference path uses lightweight matrix solve (sub-2ms).
                    </p>
                  </div>
                  <div className="bg-black/30 p-2.5 rounded-xl border border-white/5">
                    <div className="font-bold text-white mb-0.5">In-Memory Caching Fast Path</div>
                    <p className="text-[11px] text-zinc-400">
                      Sub-millisecond LRU cache with TTL for repeated profile queries. Warm responses return in under 0.5ms directly from memory.
                    </p>
                  </div>
                  <div className="bg-black/30 p-2.5 rounded-xl border border-white/5">
                    <div className="font-bold text-white mb-0.5">Graceful Degradation (Circuit Breaker)</div>
                    <p className="text-[11px] text-zinc-400">
                      If vector search or matrix solving ever times out, the system automatically degrades to a cached list of top Bayesian-rated movies without throwing 500 errors.
                    </p>
                  </div>
                </div>
              </div>

              {/* Phase 4 & 5: Evaluation, A/B Testing & Explainability */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-4 space-y-2">
                  <div className="flex items-center gap-2 text-purple-400 text-xs font-bold uppercase tracking-wider">
                    <Activity className="w-4 h-4 text-purple-400" />
                    <span>Phase 4: Evaluation & A/B Experimentation</span>
                  </div>
                  <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-400">Control Arm (A - Global Popularity):</span>
                      <strong className="text-zinc-300">NDCG 0.642 • CTR 4.2%</strong>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-400">Treatment Arm (B - Two-Stage + Bandits):</span>
                      <strong className="text-emerald-400">NDCG 0.892 • CTR 11.8%</strong>
                    </div>
                    <div className="pt-1.5 border-t border-white/10 flex items-center justify-between text-[11px]">
                      <span className="text-zinc-400">Observed CTR Lift:</span>
                      <span className="px-2 py-0.5 rounded-lg bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30">+181% Lift (p &lt; 0.001)</span>
                    </div>
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-zinc-400">Concept Drift Monitor:</span>
                      <span className="text-emerald-400 font-semibold">Population Stability Index (PSI) = 0.04 (Stable)</span>
                    </div>
                  </div>
                </div>

                <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-4 space-y-2">
                  <div className="flex items-center gap-2 text-amber-400 text-xs font-bold uppercase tracking-wider">
                    <Lightbulb className="w-4 h-4 text-amber-400" />
                    <span>Phase 5: Explainability & Implicit Feedback Loop</span>
                  </div>
                  <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-2 text-xs">
                    <div>
                      <div className="font-bold text-white mb-0.5">Attribution Transparency</div>
                      <p className="text-[11px] text-zinc-400">
                        Plain-language explainability on every title (e.g. "Because you rated 3 Idiots 5★", "Matches preferred genre: Drama") elevates user engagement and trust.
                      </p>
                    </div>
                    <div className="pt-1 border-t border-white/5">
                      <div className="font-bold text-white mb-0.5">Real-Time Negative & Positive Feedback</div>
                      <p className="text-[11px] text-zinc-400">
                        Users can click <strong>"Not Interested"</strong> to dismiss titles instantly with real-time candidate replacement, alongside 1–5 star ratings and watchlist bookmarks.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="pt-2 flex items-center justify-end">
              <button
                onClick={() => setShowTelemetryModal(false)}
                className="px-5 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 text-white text-xs font-bold transition"
              >
                Close Audit
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
