import { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { User as UserIcon, LogIn, Shield, Users } from 'lucide-react';
import { Button } from '../../components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { useAuthStore } from '../../app/store/auth';
import { useUsers } from '../../hooks/useApi';
import { toast } from 'sonner';

export function Login() {
  const { login, isAuthenticated } = useAuthStore();
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);
  const { data: users } = useUsers();

  // Redirect if already authenticated
  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  const handleLogin = async () => {
    if (!username.trim()) {
      toast.error('Please enter a username');
      return;
    }

    setLoading(true);

    try {
      // Find user in the system
      const user = users?.find(u => u.username.toLowerCase() === username.toLowerCase());

      if (user) {
        // Login with found user
        login(user);
        toast.success(`Welcome back, ${user.username}!`);
      } else {
        toast.error('User not found. Please contact an admin to create your account.');
      }
    } catch {
      toast.error('Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = () => {
    if (!users || users.length === 0) {
      toast.error('No demo users available');
      return;
    }

    const demoUser = users[0]; // Use first available user as demo
    login(demoUser);
    toast.success(`Logged in as demo user: ${demoUser.username}`);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-4">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-grid-white/10" aria-hidden="true" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative z-10 w-full max-w-sm sm:max-w-md"
      >
        {/* Logo and Title */}
        <div className="mb-6 text-center md:mb-8">
          <motion.div
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring' }}
            className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-600 to-purple-600 shadow-lg"
            aria-hidden="true"
          >
            <span className="text-2xl">&#x2696;&#xFE0F;</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="mb-2 text-2xl font-bold text-white sm:text-3xl"
          >
            RLCF Framework
          </motion.h1>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="text-sm text-slate-400 sm:text-base"
          >
            Reinforcement Learning from Community Feedback
          </motion.p>
        </div>

        {/* Login Form */}
        <Card variant="glass">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-violet-400" aria-hidden="true" />
              Sign In
            </CardTitle>
          </CardHeader>

          <CardContent>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleLogin();
              }}
              className="space-y-4"
            >
              {/* Username Field */}
              <div>
                <label
                  htmlFor="login-username"
                  className="mb-2 block text-sm font-medium text-slate-300"
                >
                  Username
                </label>
                <div className="relative">
                  <UserIcon className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" aria-hidden="true" />
                  <input
                    id="login-username"
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="w-full rounded-lg border border-slate-600 bg-slate-800/50 py-2.5 pl-10 pr-4 text-sm text-slate-200 placeholder-slate-500 transition-colors focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500 focus-visible:ring-2 focus-visible:ring-blue-500"
                    placeholder="Enter your username"
                    disabled={loading}
                    autoComplete="username"
                    aria-describedby="login-username-help"
                  />
                </div>
                <p id="login-username-help" className="mt-1 text-xs text-slate-500">
                  Enter the username of your existing RLCF account
                </p>
              </div>

              {/* Submit Buttons */}
              <div className="space-y-3">
                <Button
                  type="submit"
                  className="w-full focus-visible:ring-2 focus-visible:ring-blue-500"
                  loading={loading}
                  disabled={!username.trim()}
                >
                  <LogIn className="mr-2 h-4 w-4" aria-hidden="true" />
                  Sign In
                </Button>

                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-slate-700" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-slate-800 px-2 text-slate-400">Or</span>
                  </div>
                </div>

                <Button
                  type="button"
                  onClick={handleDemoLogin}
                  variant="secondary"
                  className="w-full focus-visible:ring-2 focus-visible:ring-blue-500"
                  disabled={!users || users.length === 0}
                >
                  Demo Login
                </Button>
              </div>
            </form>

            {/* Available Users (for development) */}
            {users && users.length > 0 && (
              <div className="mt-6 rounded-lg bg-slate-700/30 p-4">
                <div className="mb-3 flex items-center gap-2">
                  <Users className="h-4 w-4 text-slate-400" aria-hidden="true" />
                  <p className="text-sm font-medium text-slate-300">Available Users ({users.length})</p>
                </div>
                <div className="max-h-48 space-y-2 overflow-y-auto">
                  {users.slice(0, 10).map((user) => (
                    <button
                      key={user.id}
                      onClick={() => {
                        setUsername(user.username);
                        setTimeout(() => handleLogin(), 100);
                      }}
                      className="w-full rounded border border-slate-700 bg-slate-800 p-2 text-left text-sm transition-colors hover:border-slate-600 hover:bg-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-slate-200">{user.username}</span>
                        <span className="text-xs text-slate-400">
                          Authority: {user.authority_score?.toFixed(2) || '0.00'}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-6 text-center text-xs text-slate-500 md:mt-8"
        >
          Legal AI Research Framework &bull; Secure Authentication
        </motion.div>
      </motion.div>
    </div>
  );
}
